import os
import itertools
from owlready2 import get_ontology, sync_reasoner_pellet, default_world

ACTION_TAG_IMMEDIATE = "tag_immediate"
ACTION_TAG_DELAYED   = "tag_delayed"
ACTION_TAG_MINOR     = "tag_minor"
ACTION_TAG_EXPECTANT = "tag_expectant"
ACTION_OPEN_AIRWAY   = "open_airway"
ACTION_TREAT         = "treat"
ACTION_DECONTAMINATE = "decontaminate"

ALL_ACTIONS = [
    ACTION_TAG_IMMEDIATE,
    ACTION_TAG_DELAYED,
    ACTION_TAG_MINOR,
    ACTION_TAG_EXPECTANT,
    ACTION_OPEN_AIRWAY,
    ACTION_TREAT,
    ACTION_DECONTAMINATE,
]

ACTION_INDIVIDUAL_MAP = {
    ACTION_TAG_IMMEDIATE: "tagImmediate",
    ACTION_TAG_DELAYED:   "tagDelayed",
    ACTION_TAG_MINOR:     "tagMinor",
    ACTION_TAG_EXPECTANT: "tagExpectant",
    ACTION_OPEN_AIRWAY:   "openAirway",
    ACTION_TREAT:         "treat",
    ACTION_DECONTAMINATE: "decontaminate",
}

# Hazard type string (used by the patient generator / env) -> ontology
# individual name. Hazard type only governs HOW LONG decontamination
# takes (read directly off the individual, no reasoning) -- it is
# orthogonal to the reasoner-validated 32-profile lookup table, which
# only cares about the binary hasDecontaminationStatus.
HAZARD_INDIVIDUAL_MAP = {
    "chemical":     "chemicalHazard",
    "biological":   "biologicalHazard",
    "radiological": "radiologicalHazard",
    "nuclear":      "nuclearHazard",
}

# Reverse lookup: ontology individual name -> action string
_REVERSE_ACTION_MAP = {v: k for k, v in ACTION_INDIVIDUAL_MAP.items()}

CATEGORY_TO_ACTION = {
    "Minor":     ACTION_TAG_MINOR,
    "Delayed":   ACTION_TAG_DELAYED,
    "Immediate": ACTION_TAG_IMMEDIATE,
    "Expectant": ACTION_TAG_EXPECTANT,
}

SEVERITY_BY_CATEGORY = {"Minor": 0, "Delayed": 1, "Immediate": 2, "Expectant": 3}


def _load_ontology_raw(owl_path: str):
    abs_path = os.path.abspath(owl_path)
    return get_ontology(f"file://{abs_path}").load()


_ONTO_PATH_CACHE = {}


def load_ontology(owl_path: str):
    """Loads the ontology and eagerly warms the reasoner-backed lookup
    table cache (runs Pellet once over all 32 profiles, executing both
    the OWL-DL equivalentClass classification and the SWRL rule set)."""
    onto = _load_ontology_raw(owl_path)
    _ONTO_PATH_CACHE[id(onto)] = owl_path
    _table_for(owl_path)
    return onto


def _owl_path_for_onto(onto):
    if id(onto) in _ONTO_PATH_CACHE:
        return _ONTO_PATH_CACHE[id(onto)]
    raise RuntimeError(
        "Ontology was not loaded via load_ontology(); cannot resolve its "
        "path for the lookup table cache."
    )


def _get(onto, *names):
    for name in names:
        ind = onto[name]
        if ind is not None:
            return ind
    raise ValueError(
        f"None of {names} found in ontology. "
        f"Available: {[i.name for i in onto.individuals()]}"
    )


def _get_class_comment(onto, class_name):
    cls = onto[class_name]
    if cls is not None and hasattr(cls, "comment") and cls.comment:
        return cls.comment[0]
    return f"Ontology class {class_name}: no comment."


def _get_action_comment(onto, action_individual_name):
    ind = onto[action_individual_name]
    if ind is not None and hasattr(ind, "comment") and ind.comment:
        return ind.comment[0]
    return f"Action {action_individual_name}."


# ---------------------------------------------------------------------------
# Reasoner-backed lookup table compilation (runs Pellet exactly ONCE)
#
# Pellet (unlike HermiT) executes SWRL rules in addition to standard OWL-DL
# classification, so this is the only place in the pipeline where the
# ontology's `equivalentClass` axioms AND its swrl:Imp rules are both
# actually fired. The two are read independently and cross-checked below.
# ---------------------------------------------------------------------------

_TABLE_CACHE = {}


def _build_lookup_table(owl_path: str):
    onto = _load_ontology_raw(owl_path)

    profiles = list(itertools.product([True, False], repeat=5))
    individuals = {}
    with onto:
        for i, profile in enumerate(profiles):
            amb, br, pulse, cmd, dec = profile
            p = onto.Patient(f"_lut_p{i}")
            p.hasAmbulationStatus = [onto.ambulatory if amb else onto.nonAmbulatory]
            p.hasRespiratoryStatus = [onto.breathing if br else onto.notBreathing]
            p.hasPulseStatus = [onto.pulsePresent if pulse else onto.pulseAbsent]
            p.hasMentalStatus = [onto.followsCommands if cmd else onto.doesNotFollowCommands]
            p.hasDecontaminationStatus = [
                onto.decontaminated if dec else onto.notDecontaminated
            ]
            individuals[profile] = p

        # infer_property_values fires SWRL atoms with object-property heads
        # (treatmentRequired); infer_data_property_values fires SWRL atoms
        # with datatype-property heads (hasSeverityScore).
        sync_reasoner_pellet(
            infer_property_values=True,
            infer_data_property_values=True,
            debug=0,
        )

    tag_classes = {
        "Minor": onto.Minor,
        "Delayed": onto.Delayed,
        "Immediate": onto.Immediate,
        "Expectant": onto.Expectant,
    }
    comments = {name: _get_class_comment(onto, name) for name in tag_classes}
    decon_comment = _get_class_comment(onto, "RequiresDecontaminationFirst")

    table = {}
    for profile, p in individuals.items():
        amb, br, pulse, cmd, dec = profile

        # --- Path 1: OWL-DL classification via equivalentClass axioms ---
        inferred = [name for name, cls in tag_classes.items() if cls in p.INDIRECT_is_a]
        if len(inferred) != 1:
            raise RuntimeError(
                f"Reasoner did not assign exactly one triage category to "
                f"profile {profile}: got {inferred}. Check the ontology's "
                f"equivalentClass definitions for overlap/gaps."
            )
        category = inferred[0]
        requires_decon = onto.RequiresDecontaminationFirst in p.INDIRECT_is_a

        # --- Path 2: SWRL rule inference (treatmentRequired, hasSeverityScore) ---
        swrl_actions = [
            _REVERSE_ACTION_MAP[a.name] for a in p.treatmentRequired
            if hasattr(a, "name") and a.name in _REVERSE_ACTION_MAP
        ]
        swrl_severities = list(p.hasSeverityScore)

        if len(swrl_actions) != 1:
            raise RuntimeError(
                f"SWRL rules did not infer exactly one treatmentRequired "
                f"value for profile {profile}: got {swrl_actions}. Check "
                f"SWRL rule coverage / isRuleEnabled flags in the ontology."
            )
        if len(swrl_severities) != 1:
            raise RuntimeError(
                f"SWRL rules did not infer exactly one hasSeverityScore "
                f"value for profile {profile}: got {swrl_severities}."
            )

        swrl_action = swrl_actions[0]
        swrl_severity = int(swrl_severities[0])  # Pellet returns xsd:integer as float

        # --- Cross-check: the two independent reasoning paths must agree ---
        expected_action = CATEGORY_TO_ACTION[category]
        expected_severity = SEVERITY_BY_CATEGORY[category]
        if swrl_action != expected_action or swrl_severity != expected_severity:
            raise RuntimeError(
                f"SWRL/DL disagreement for profile {profile}: DL "
                f"classification gives category={category} "
                f"(expects action={expected_action}, severity={expected_severity}), "
                f"but SWRL inferred action={swrl_action}, severity={swrl_severity}."
            )

        valid = set()
        if category == "Minor":
            valid.add(ACTION_TAG_MINOR)
        elif category == "Expectant":
            # Tightened by design choice: over-triaging an Expectant
            # patient to tag_immediate misallocates scarce resources
            # away from patients who could actually benefit, so it is
            # now contraindicated -- tag_expectant is the only valid
            # tag for this category. See the updated Expectant class
            # comment in the ontology for the patient-facing reason
            # surfaced by check_action().
            valid.add(ACTION_TAG_EXPECTANT)
        else:
            valid.add(ACTION_TAG_IMMEDIATE)
            valid.add(ACTION_TAG_EXPECTANT)
            if category == "Delayed":
                valid.add(ACTION_TAG_DELAYED)
        valid.add(ACTION_OPEN_AIRWAY)
        valid.add(ACTION_DECONTAMINATE)  # never medically contraindicated
        if not requires_decon:
            valid.add(ACTION_TREAT)

        table[profile] = {
            "category": category,
            "severity": swrl_severity,           # genuinely SWRL-inferred now
            "recommended_action": swrl_action,    # genuinely SWRL-inferred now
            "requires_decon": requires_decon,
            "valid_actions": valid,
            "category_comment": comments[category],
            "decon_comment": decon_comment if requires_decon else None,
        }
    return table


def _table_for(owl_path: str):
    abs_path = os.path.abspath(owl_path)
    if abs_path not in _TABLE_CACHE:
        _TABLE_CACHE[abs_path] = _build_lookup_table(owl_path)
    return _TABLE_CACHE[abs_path]


def create_patient(onto, patient_id: str, ambulatory: bool,
                    breathing: bool, pulse: bool, follows_commands: bool,
                    decontaminated: bool = True, hazard_type: str = None):
    """
    hazard_type: one of "chemical", "biological", "radiological", "nuclear",
    or None. Only affects how many `decontaminate` actions are needed
    (via get_decontamination_duration) -- it does not change triage
    category or KG-validity of any action.
    """

    amb_ind = _get(onto, "ambulatory", "Ambulatory")
    namb_ind = _get(onto, "nonAmbulatory", "NonAmbulatory")
    br_ind = _get(onto, "breathing", "Breathing")
    nbr_ind = _get(onto, "notBreathing", "NotBreathing")
    pp_ind = _get(onto, "pulsePresent", "PulsePresent")
    pa_ind = _get(onto, "pulseAbsent", "PulseAbsent")
    fc_ind = _get(onto, "followsCommands", "FollowsCommands")
    dfc_ind = _get(onto, "doesNotFollowCommands", "DoesNotFollowCommands")
    dec_ind = _get(onto, "decontaminated", "Decontaminated")
    ndec_ind = _get(onto, "notDecontaminated", "NotDecontaminated")

    with onto:
        p = onto.Patient(patient_id)
        p.hasAmbulationStatus = [amb_ind if ambulatory else namb_ind]
        p.hasRespiratoryStatus = [br_ind if breathing else nbr_ind]
        p.hasPulseStatus = [pp_ind if pulse else pa_ind]
        p.hasMentalStatus = [fc_ind if follows_commands else dfc_ind]
        p.hasDecontaminationStatus = [dec_ind if decontaminated else ndec_ind]
        p.is_decontaminated = decontaminated

        if hazard_type is not None:
            hazard_ind_name = HAZARD_INDIVIDUAL_MAP.get(hazard_type)
            if hazard_ind_name is None:
                raise ValueError(
                    f"Unknown hazard_type '{hazard_type}'. Expected one of "
                    f"{list(HAZARD_INDIVIDUAL_MAP)}."
                )
            hazard_ind = _get(onto, hazard_ind_name)
            p.hasHazardType = [hazard_ind]
    return p


def get_decontamination_duration(onto, patient, default=1):
    """
    Reads how many `decontaminate` actions this patient's hazard type
    requires, via a direct property hop onto the hazard individual.
    This is plain data lookup, not reasoning -- hazard severity isn't
    something the DL/SWRL layer needs to classify a patient.
    Returns `default` if the patient has no asserted hazard type.
    """
    if not hasattr(patient, "hasHazardType") or not patient.hasHazardType:
        return default
    hazard = patient.hasHazardType[0]
    if hasattr(hazard, "hasDecontaminationDuration") and hazard.hasDecontaminationDuration:
        return int(hazard.hasDecontaminationDuration[0])
    return default


def mark_decontaminated(onto, patient):
    """
    Mutates the patient's hasDecontaminationStatus to Decontaminated.
    Called by the environment once enough `decontaminate` actions have
    accumulated to satisfy the hazard type's required duration. Because
    hasDecontaminationStatus is already one of the 5 dimensions the
    lookup table is keyed on, the next check_action()/get_valid_actions()
    call against this patient will automatically reflect the change --
    no new reasoning call needed.
    """
    dec_ind = _get(onto, "decontaminated", "Decontaminated")
    with onto:
        patient.hasDecontaminationStatus = [dec_ind]


def _profile_key(patient):
    onto = patient.namespace.ontology
    amb_ind = _get(onto, "ambulatory", "Ambulatory")
    br_ind = _get(onto, "breathing", "Breathing")
    pp_ind = _get(onto, "pulsePresent", "PulsePresent")
    fc_ind = _get(onto, "followsCommands", "FollowsCommands")
    dec_ind = _get(onto, "decontaminated", "Decontaminated")

    amb = amb_ind in patient.hasAmbulationStatus
    br = br_ind in patient.hasRespiratoryStatus
    pls = pp_ind in patient.hasPulseStatus
    cmd = fc_ind in patient.hasMentalStatus
    dec = dec_ind in patient.hasDecontaminationStatus
    return (amb, br, pls, cmd, dec), onto.base_iri


def check_action(onto, patient, proposed_action: str):
    """
    Validates proposed RL action against the reasoner-derived lookup table
    (see _build_lookup_table). Returns (is_valid: bool, justification: str).
    """
    key, base_iri = _profile_key(patient)
    table = _table_for(_owl_path_for_onto(onto))
    entry = table[key]

    if proposed_action in entry["valid_actions"]:
        action_ind = ACTION_INDIVIDUAL_MAP.get(proposed_action, proposed_action)
        return True, f"[KG check passed] {_get_action_comment(onto, action_ind)}"

    if proposed_action == ACTION_TREAT and entry["requires_decon"]:
        return False, f"[CBRN Rule 5] {entry['decon_comment']}"

    return False, (
        f"[KG: classified {entry['category']}] {entry['category_comment']} "
        f"Action '{proposed_action}' contraindicated."
    )


def get_valid_actions(onto, patient):
    """Returns all KG-permitted actions for this patient."""
    key, _ = _profile_key(patient)
    table = _table_for(_owl_path_for_onto(onto))
    return list(table[key]["valid_actions"])


def get_contraindicated_actions(onto, proposed_action: str):
    ind_name = ACTION_INDIVIDUAL_MAP.get(proposed_action)
    if not ind_name:
        return []
    ind = onto[ind_name]
    if ind is None or not hasattr(ind, "contraindicates"):
        return []
    reverse_map = {v: k for k, v in ACTION_INDIVIDUAL_MAP.items()}
    return [reverse_map[c.name] for c in ind.contraindicates
            if hasattr(c, "name") and c.name in reverse_map]


def _infer_severity(onto, patient):
    key, _ = _profile_key(patient)
    table = _table_for(_owl_path_for_onto(onto))
    return table[key]["severity"]


def get_kg_recommended_action(onto, patient):
    key, _ = _profile_key(patient)
    table = _table_for(_owl_path_for_onto(onto))
    return table[key]["recommended_action"]


def get_full_kg_trace(onto, patient):
    key, _ = _profile_key(patient)
    table = _table_for(_owl_path_for_onto(onto))
    entry = table[key]
    severity_labels = {0: "minor", 1: "stable", 2: "serious", 3: "critical"}

    lines = []
    lines.append(f"  Reasoner-inferred category : {entry['category']}")
    lines.append(f"  KG recommended action      : {entry['recommended_action']}")
    lines.append(f"  KG severity score          : {entry['severity']} "
                  f"({severity_labels.get(entry['severity'], '?')})")
    lines.append(f"  Decontaminated             : {not entry['requires_decon']}")
    lines.append(f"  Action constraints:")

    for action in ALL_ACTIONS:
        valid, reason = check_action(onto, patient, action)
        status = "PERMITTED" if valid else "BLOCKED "
        contra = get_contraindicated_actions(onto, action)
        contra_str = f"  [contraindicates: {', '.join(contra)}]" if contra else ""
        lines.append(f"    {action:<20} [{status}] {reason[:100]}{contra_str}")

    return "\n".join(lines)