import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kg.constraint_guard import (
    load_ontology, create_patient, get_kg_recommended_action,
)

OWL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ontology", "triage_v2.rdf"
)

SYN_STARTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "syn_starts_all_scenarios.json",
)

ACTION_TO_GOLD_COLOR = {
    "tag_minor":     "Green",
    "tag_delayed":   "Yellow",
    "tag_immediate": "Red",
    "tag_expectant": "Black",
}

ACTION_TO_CATEGORY_LABEL = {
    "tag_minor":     "Minor",
    "tag_delayed":   "Delayed",
    "tag_immediate": "Immediate",
    "tag_expectant": "Expectant",
}


def _extract_breathing(respirations):
    """Locked conversion rule (see SYN_STARTS_CONVERSION_RULE.md):
    breathing = (rate > 0) if a rate is given; else
    breathing = breathing_after_maneuver if that field is given
    (the protocol-relevant status once airway-opening has been
    attempted); else not derivable."""
    if respirations is None:
        return None
    if "rate" in respirations and isinstance(respirations["rate"], (int, float)):
        return respirations["rate"] > 0
    if "breathing_after_maneuver" in respirations and isinstance(
        respirations["breathing_after_maneuver"], bool
    ):
        return respirations["breathing_after_maneuver"]
    return None

def convert_case(case):
    """Returns (ambulatory, breathing, pulse, follows_commands, respiratory_rate) or None
    if any of the four cannot be mechanically derived under the locked
    conversion rule."""
    v = case["vitals_info"]

    if "can_walk" not in v or not isinstance(v["can_walk"], bool):
        return None
    ambulatory = v["can_walk"]

    respirations = v.get("respirations")
    breathing = _extract_breathing(respirations)
    if breathing is None:
        return None

    # Extract raw respiratory rate if available
    respiratory_rate = None
    if respirations is not None and "rate" in respirations and isinstance(
        respirations["rate"], (int, float)
    ):
        respiratory_rate = int(respirations["rate"])

    perfusion = v.get("perfusion")
    if perfusion is None or "radial_pulse_present" not in perfusion or not isinstance(
        perfusion["radial_pulse_present"], bool
    ):
        return None
    pulse = perfusion["radial_pulse_present"]

    mental_status = v.get("mental_status")
    if mental_status is None or "obeys_commands" not in mental_status or not isinstance(
        mental_status["obeys_commands"], bool
    ):
        return None
    follows_commands = mental_status["obeys_commands"]

    return ambulatory, breathing, pulse, follows_commands, respiratory_rate

def evaluate_external_validation(owl_path: str = OWL_PATH,
                                   json_path: str = SYN_STARTS_JSON_PATH):
    """
    Loads the real ontology (running Pellet), converts every Syn-STARTS
    case with all four features mechanically derivable, classifies each
    through the real reasoner-backed pipeline, and returns the full list
    of per-case results plus the excluded count.
    """
    onto = load_ontology(owl_path)
    with open(json_path, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    excluded = 0

    for case in cases:
        converted = convert_case(case)
        if converted is None:
            excluded += 1
            continue
        ambulatory, breathing, pulse, follows_commands, respiratory_rate = converted

        patient = create_patient(
            onto,
            patient_id=case["scenario_id"],
            ambulatory=ambulatory,
            breathing=breathing,
            pulse=pulse,
            follows_commands=follows_commands,
            decontaminated=True,
            respiratory_rate=respiratory_rate,
        )

        recommended_action = get_kg_recommended_action(onto, patient)
        predicted_category = ACTION_TO_CATEGORY_LABEL[recommended_action]
        predicted_color = ACTION_TO_GOLD_COLOR[recommended_action]
        gold = case["triage_tag"].capitalize()  # normalizes "BLACK" -> "Black"
        match = predicted_color == gold

        results.append({
            "scenario_id": case["scenario_id"],
            "gold": gold,
            "predicted_category": predicted_category,
            "predicted_color": predicted_color,
            "match": match,
            "ambulatory": ambulatory,
            "breathing": breathing,
            "pulse": pulse,
            "follows_commands": follows_commands,
        })

    return results, excluded


if __name__ == "__main__":
    results, excluded = evaluate_external_validation()

    print("=== External validation against the Syn-STARTS structured "
          "benchmark ===")
    print("(classification computed by the real Pellet-backed reasoner "
          "pipeline -- no transcription)\n")

    total = len(results)
    n_match = sum(r["match"] for r in results)
    print(f"Total cases in corpus: {total + excluded}")
    print(f"Excluded (could not mechanically derive all 4 features): "
          f"{excluded}")
    print(f"Scoreable: {total}")
    print(f"Overall agreement: {n_match}/{total} = "
          f"{100 * n_match / total:.1f}%\n")

    by_tag = {}
    for r in results:
        by_tag.setdefault(r["gold"], []).append(r)
    print("Agreement by gold category:")
    for tag in ["Green", "Yellow", "Red", "Black"]:
        if tag in by_tag:
            rs = by_tag[tag]
            m = sum(r["match"] for r in rs)
            print(f"  {tag:8}: {m}/{len(rs)} = {100 * m / len(rs):.1f}%")

    print("\nFirst 10 disagreements:")
    shown = 0
    for r in results:
        if not r["match"] and shown < 10:
            print(f"  {r['scenario_id']}: gold={r['gold']:7} "
                  f"predicted={r['predicted_color']:7} "
                  f"({r['predicted_category']}) | "
                  f"amb={r['ambulatory']} br={r['breathing']} "
                  f"pulse={r['pulse']} cmd={r['follows_commands']}")
            shown += 1

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                            "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "syn_starts_validation_results.csv")
    import csv
    fieldnames = ["scenario_id", "gold", "predicted_category",
                  "predicted_color", "match", "ambulatory", "breathing",
                  "pulse", "follows_commands"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved {len(results)} results to {out_path}")