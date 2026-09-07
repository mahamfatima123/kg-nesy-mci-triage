import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from owlready2 import get_ontology, sync_reasoner_pellet

OWL_PATH = os.path.join(os.path.dirname(__file__), "..", "ontology", "OWL_Ontology.rdf")
BOUNDARY_RATES = [29, 30, 31]   # one below, the cutoff itself, one above

CATEGORY_TO_ACTION = {
    "Minor": "tagMinor", "Delayed": "tagDelayed",
    "Immediate": "tagImmediate", "Expectant": "tagExpectant",
}


def test_tachypnea_boundary():
    onto = get_ontology(f"file://{os.path.abspath(OWL_PATH)}").load()
    tag_classes = {
        "Minor": onto.Minor, "Delayed": onto.Delayed,
        "Immediate": onto.Immediate, "Expectant": onto.Expectant,
    }

    profiles = list(itertools.product([True, False], repeat=4))  # amb, pulse, cmd, dec
    individuals = {}

    with onto:
        idx = 0
        for amb, pulse, cmd, dec in profiles:
            for rate in BOUNDARY_RATES:
                p = onto.Patient(f"_boundary_p{idx}"); idx += 1
                p.hasAmbulationStatus      = [onto.ambulatory      if amb   else onto.nonAmbulatory]
                p.hasRespiratoryStatus     = [onto.breathing]
                p.hasPulseStatus           = [onto.pulsePresent    if pulse else onto.pulseAbsent]
                p.hasMentalStatus          = [onto.followsCommands if cmd   else onto.doesNotFollowCommands]
                p.hasDecontaminationStatus = [onto.decontaminated  if dec   else onto.notDecontaminated]
                p.hasRespiratoryRate       = [rate]
                individuals[(amb, pulse, cmd, dec, rate)] = p

        sync_reasoner_pellet(infer_property_values=True,
                              infer_data_property_values=True, debug=0)

    n = len(individuals)
    print(f"Running boundary check on {n} individuals "
          f"(16 vital combinations x {len(BOUNDARY_RATES)} rates: {BOUNDARY_RATES})...")

    errors = []
    for profile, p in individuals.items():
        amb, pulse, cmd, dec, rate = profile
        inferred = [name for name, cls in tag_classes.items() if cls in p.INDIRECT_is_a]
        if len(inferred) != 1:
            errors.append((profile, f"DL classification not unique: {inferred}"))
            continue
        category = inferred[0]
        swrl_actions = [a.name for a in p.treatmentRequired if hasattr(a, "name")]
        expected = CATEGORY_TO_ACTION[category]
        if len(swrl_actions) != 1 or swrl_actions[0] != expected:
            errors.append((profile, f"DL={category} but SWRL treatmentRequired={swrl_actions}"))

        # The one profile where the boundary should visibly flip the
        # category: stable, non-ambulatory patient with pulse, breathing,
        # following commands -- Delayed at rate<=30, Immediate at rate>30.
        if not amb and pulse and cmd and category not in ("Delayed", "Immediate"):
            errors.append((profile, f"expected Delayed/Immediate at this rate, got {category}"))
        if not amb and pulse and cmd:
            if rate <= 30 and category != "Delayed":
                errors.append((profile, f"rate={rate} (<=30) should be Delayed, got {category}"))
            if rate > 30 and category != "Immediate":
                errors.append((profile, f"rate={rate} (>30) should be Immediate, got {category}"))

    if errors:
        print(f"\n{len(errors)} FAILURE(S):")
        for e in errors:
            print("  ", e)
        raise AssertionError(f"{len(errors)} boundary disagreement(s) found -- see above.")

    print(f"PASS: DL and SWRL agree on all {n} boundary profiles.")
    print("PASS: rate=30 correctly classified Delayed (not yet tachypneic); "
          "rate=31 correctly classified Immediate (tachypneic) -- confirms "
          "the threshold fires on strict '>30', not '>=30'.")


if __name__ == "__main__":
    test_tachypnea_boundary()
