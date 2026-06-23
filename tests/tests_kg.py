import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kg.constraint_guard import (
    load_ontology, create_patient, check_action, get_valid_actions,
    ACTION_TAG_IMMEDIATE, ACTION_TAG_DELAYED,
    ACTION_TAG_MINOR, ACTION_TAG_EXPECTANT, ACTION_OPEN_AIRWAY, ACTION_TREAT
)

OWL_PATH = "ontology/triage_v2.rdf"

def make(onto, pid, ambulatory, breathing, pulse, follows_commands,
         decontaminated=True):
    return create_patient(onto, pid, ambulatory=ambulatory, breathing=breathing,
                          pulse=pulse, follows_commands=follows_commands,
                          decontaminated=decontaminated)

def test_all():
    onto = load_ontology(OWL_PATH)
    print("Ontology loaded. Running 8 patient scenario tests...\n")

    # --- Scenario 1: Minor category -> ONLY tag_minor is valid ---
    p1 = make(onto, "p1", True,  True,  True,  True)
    assert check_action(onto, p1, ACTION_TAG_MINOR)[0]
    assert not check_action(onto, p1, ACTION_TAG_IMMEDIATE)[0]
    assert not check_action(onto, p1, ACTION_TAG_DELAYED)[0]
    assert not check_action(onto, p1, ACTION_TAG_EXPECTANT)[0]
    print("PASS scenario 1: ambulatory patient — START Rule 1")

    # --- Scenario 2: Immediate category (no breathing, pulse present) ---
    # Allowed: tag_immediate, tag_expectant. Blocked: tag_minor, tag_delayed.
    p2 = make(onto, "p2", False, False, True,  False)
    assert check_action(onto, p2, ACTION_TAG_IMMEDIATE)[0]
    assert check_action(onto, p2, ACTION_TAG_EXPECTANT)[0]
    assert not check_action(onto, p2, ACTION_TAG_MINOR)[0]
    assert not check_action(onto, p2, ACTION_TAG_DELAYED)[0]
    print("PASS scenario 2: not breathing patient — START Rule 2")

    # --- Scenario 3: Immediate category (breathing, no pulse) ---
    p3 = make(onto, "p3", False, True,  False, False)
    assert check_action(onto, p3, ACTION_TAG_IMMEDIATE)[0]
    assert check_action(onto, p3, ACTION_TAG_EXPECTANT)[0]
    assert not check_action(onto, p3, ACTION_TAG_MINOR)[0]
    assert not check_action(onto, p3, ACTION_TAG_DELAYED)[0]
    print("PASS scenario 3: no pulse patient — START Rule 3")

    # --- Scenario 4: Immediate category (breathing, pulse, no commands) ---
    p4 = make(onto, "p4", False, True,  True,  False)
    assert check_action(onto, p4, ACTION_TAG_IMMEDIATE)[0]
    assert check_action(onto, p4, ACTION_TAG_EXPECTANT)[0]
    assert not check_action(onto, p4, ACTION_TAG_MINOR)[0]
    assert not check_action(onto, p4, ACTION_TAG_DELAYED)[0]
    print("PASS scenario 4: no commands patient — START Rule 4")

    # --- Scenario 5: Delayed category -> all of delayed/immediate/expectant
    # are allowed, only tag_minor is blocked ---
    p5 = make(onto, "p5", False, True,  True,  True)
    assert check_action(onto, p5, ACTION_TAG_DELAYED)[0]
    assert check_action(onto, p5, ACTION_TAG_IMMEDIATE)[0]
    assert check_action(onto, p5, ACTION_TAG_EXPECTANT)[0]
    assert not check_action(onto, p5, ACTION_TAG_MINOR)[0]
    print("PASS scenario 5: stable patient can be Delayed")

    # --- Scenario 6: same patient as p2, checked via get_valid_actions()
    # instead of one-action-at-a-time -- same conclusion, different method ---
    valid_actions = get_valid_actions(onto, p2)
    assert ACTION_TAG_DELAYED not in valid_actions
    assert ACTION_TAG_MINOR   not in valid_actions
    assert ACTION_TAG_IMMEDIATE in valid_actions
    assert ACTION_TAG_EXPECTANT in valid_actions
    print(f"PASS scenario 6: valid actions for non-breathing = {valid_actions}")

    # --- Scenario 7: CBRN Rule 5 -- treat blocked while contaminated,
    # permitted once clean; open_airway unaffected either way ---
    p7 = make(onto, "p7", False, True, True, True, decontaminated=False)
    valid, reason = check_action(onto, p7, ACTION_TREAT)
    assert not valid, "FAIL 7a"
    assert "CBRN Rule 5" in reason, f"FAIL 7b: {reason}"
    assert check_action(onto, p7, ACTION_OPEN_AIRWAY)[0], "FAIL 7c"
    p7c = make(onto, "p7_clean", False, True, True, True, decontaminated=True)
    assert check_action(onto, p7c, ACTION_TREAT)[0], "FAIL 7d"
    print("PASS scenario 7: CBRN Rule 5 — treat blocked when contaminated, permitted when clean")

    # --- Scenario 8: Expectant category -> only tag_expectant is valid.
    # tag_immediate is now contraindicated (hand-added tightening rule);
    # tag_minor and tag_delayed were already blocked by base START logic.
    p8 = make(onto, "p8", False, False, False, True)
    valid, reason = check_action(onto, p8, ACTION_TAG_IMMEDIATE)
    assert not valid, f"FAIL 8a: tag_immediate should be contraindicated for Expectant, got: {reason}"
    assert check_action(onto, p8, ACTION_TAG_EXPECTANT)[0], "FAIL 8b"
    assert not check_action(onto, p8, ACTION_TAG_MINOR)[0], "FAIL 8e"
    assert not check_action(onto, p8, ACTION_TAG_DELAYED)[0], "FAIL 8f"
    valid_actions_p8 = get_valid_actions(onto, p8)
    assert ACTION_TAG_IMMEDIATE not in valid_actions_p8, "FAIL 8c"
    assert ACTION_TAG_EXPECTANT in valid_actions_p8, "FAIL 8d"
    tag_actions_p8 = [a for a in valid_actions_p8 if a.startswith("tag_")]
    print(f"PASS scenario 8: Expectant patient — tag_immediate now contraindicated, "
          f"valid tag actions = {tag_actions_p8}")

    print("\nAll 8 scenarios passed.")

if __name__ == "__main__":
    test_all()