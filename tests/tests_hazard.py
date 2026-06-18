import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kg.constraint_guard import (
    load_ontology, create_patient, check_action,
    get_decontamination_duration, mark_decontaminated,
    ACTION_DECONTAMINATE, ACTION_TREAT
)

OWL_PATH = "ontology/triage_v2.rdf"


def test_hazard_wiring():
    onto = load_ontology(OWL_PATH)
    print("Ontology loaded (with hazard types). Running checks...\n")

    # Nuclear hazard should require 3 decontaminate actions
    p_nuclear = create_patient(onto, "hazard_test_nuclear",
                                ambulatory=False, breathing=True, pulse=True,
                                follows_commands=True, decontaminated=False,
                                hazard_type="nuclear")
    duration = get_decontamination_duration(onto, p_nuclear)
    print(f"Nuclear hazard decon duration: {duration} (expect 3)")
    assert duration == 3, f"FAIL: expected 3, got {duration}"

    # Chemical hazard should require 1
    p_chem = create_patient(onto, "hazard_test_chemical",
                             ambulatory=False, breathing=True, pulse=True,
                             follows_commands=True, decontaminated=False,
                             hazard_type="chemical")
    duration = get_decontamination_duration(onto, p_chem)
    print(f"Chemical hazard decon duration: {duration} (expect 1)")
    assert duration == 1, f"FAIL: expected 1, got {duration}"

    # decontaminate should always be KG-valid, regardless of hazard/category
    valid, reason = check_action(onto, p_nuclear, ACTION_DECONTAMINATE)
    print(f"decontaminate valid on contaminated patient: {valid} | {reason}")
    assert valid

    # treat should be blocked before decontamination...
    valid, reason = check_action(onto, p_nuclear, ACTION_TREAT)
    print(f"treat valid before decon: {valid} | {reason}")
    assert not valid

    # ...and permitted after mark_decontaminated() flips the status
    mark_decontaminated(onto, p_nuclear)
    valid, reason = check_action(onto, p_nuclear, ACTION_TREAT)
    print(f"treat valid after decon: {valid} | {reason}")
    assert valid

    # A patient with no hazard_type at all should fall back to the default
    p_none = create_patient(onto, "hazard_test_none",
                             ambulatory=False, breathing=True, pulse=True,
                             follows_commands=True, decontaminated=True)
    duration = get_decontamination_duration(onto, p_none)
    print(f"No-hazard-type fallback duration: {duration} (expect default 1)")
    assert duration == 1

    print("\nAll hazard-type wiring checks passed.")


if __name__ == "__main__":
    test_hazard_wiring()