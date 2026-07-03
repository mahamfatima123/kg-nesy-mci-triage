import json
import os
import sys
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kg.constraint_guard import load_ontology, create_patient, get_kg_recommended_action
from src.eval.syn_starts_conversion import (
    convert_case, ACTION_TO_GOLD_COLOR, ACTION_TO_CATEGORY_LABEL,
    _extract_breathing, _extract_pulse,
)

OWL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ontology", "triage_v2.rdf"
)

SYN_STARTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data",
    "syn_starts_all_scenarios.json",
)


def diagnose_exclusion_reason(case):
    """Like convert_case, but for excluded cases only: returns which
    single field was the first blocker (matching convert_case's
    short-circuit order), or 'scoreable' if it would have passed."""
    v = case["vitals_info"]

    if "can_walk" not in v or not isinstance(v["can_walk"], bool):
        return "ambulatory"
    breathing = _extract_breathing(v.get("respirations"))
    if breathing is None:
        return "breathing"
    pulse = _extract_pulse(v.get("perfusion"))
    if pulse is None:
        return "pulse"
    mental_status = v.get("mental_status")
    if mental_status is None or "obeys_commands" not in mental_status or not isinstance(
        mental_status["obeys_commands"], bool
    ):
        return "mental_status"
    return "scoreable"

def plot_exclusion_reasons(cases, out_path="results/syn_starts_exclusion_reasons.png"):
    """Stacked bar chart: for each gold category, how many of its 500
    scenarios are scoreable vs. excluded, broken down by WHICH field
    was missing. Saved to the results folder for the repo README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Libertinus Serif", "Linux Libertine",   # matches the paper, if installed as a system font
        "Times New Roman", "Cambria", "Georgia",  # close visual matches, pre-installed on Windows
        "Liberation Serif", "DejaVu Serif",       # further fallbacks
    ]
    plt.rcParams["mathtext.fontset"] = "stix"     # matches serif math styling

    categories = ["Green", "Yellow", "Red", "Black"]
    fields = ["scoreable", "breathing", "pulse", "mental_status", "ambulatory"]
    colors = {
        "scoreable": "#4C9A72", "breathing": "#7FA6D6",
        "pulse": "#F0A868", "mental_status": "#C9C9C9", "ambulatory": "#D98880",
    }
    labels = {
        "scoreable": "Scoreable", "breathing": "Excluded — breathing missing",
        "pulse": "Excluded — pulse missing",
        "mental_status": "Excluded — mental status missing",
        "ambulatory": "Excluded — ambulatory missing",
    }

    counts = {cat: {f: 0 for f in fields} for cat in categories}
    for case in cases:
        cat = case["triage_tag"].capitalize()
        if cat not in counts:
            continue
        reason = diagnose_exclusion_reason(case)
        counts[cat][reason] += 1

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bar_height = 0.5
    y_pos = np.arange(len(categories)) * 1.0
    left = np.zeros(len(categories))

    for field in fields:
        vals = np.array([counts[c][field] for c in categories], dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(y_pos, vals, left=left, height=bar_height,
                color=colors[field], label=labels[field],
                edgecolor="white", linewidth=0.6)
        if field == "scoreable":
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v > 25:
                    ax.text(l + v / 2, y_pos[yi], f"{int(v)}", ha="center", va="center",
                            color="white", fontsize=10)
                elif v > 0:
                    ax.text(l + v + 6, y_pos[yi], f"{int(v)}", ha="left", va="center",
                            color=colors["scoreable"], fontsize=9.5)
        left += vals

    for yi, cat in enumerate(categories):
        pct = 100 * counts[cat]["scoreable"] / 500
        ax.text(505, y_pos[yi], f"{pct:.0f}% scoreable", ha="left", va="center",
                fontsize=9.5, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 620)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel("Scenarios (out of 500 per category)", fontsize=10.5,
                  color="#444444", labelpad=8)

    ax.set_title("Why Syn-STARTS scenarios are excluded", fontsize=14,
                 fontweight="bold", color="#222222", pad=34, loc="left")
    ax.text(0, 1.10, "Missing field, by gold triage category",
            transform=ax.transAxes, fontsize=10.5, color="#666666", ha="left")

    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0, colors="#444444")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9.5, handlelength=1.2, handleheight=1.2,
              columnspacing=1.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exclusion-reason chart to {out_path}")


def evaluate_external_validation(owl_path: str = OWL_PATH,
                                   json_path: str = SYN_STARTS_JSON_PATH):
    """
    Loads the real ontology (running Pellet), converts every Syn-STARTS
    case with all four features mechanically derivable (via the locked
    rule in src/eval/syn_starts_conversion.py), classifies each through
    the real reasoner-backed pipeline, and returns the full list of
    per-case results plus the excluded count.
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
        gold = case["triage_tag"].capitalize()
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

    with open(SYN_STARTS_JSON_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    print("=== External validation against the Syn-STARTS structured "
          "benchmark ===")
    print("(classification computed by the real Pellet-backed reasoner "
          "pipeline -- no transcription)\n")

    total = len(results)
    n_match = sum(r["match"] for r in results)
    print(f"Total cases in corpus: {total + excluded}")
    print(f"Excluded (could not mechanically derive all 4 features): {excluded}")
    print(f"Scoreable: {total}")
    print(f"Overall agreement: {n_match}/{total} = {100 * n_match / total:.1f}%\n")

    by_tag = {}
    for r in results:
        by_tag.setdefault(r["gold"], []).append(r)
    print("Agreement by gold category:")
    for tag in ["Green", "Yellow", "Red", "Black"]:
        if tag in by_tag:
            rs = by_tag[tag]
            m = sum(r["match"] for r in rs)
            print(f"  {tag:8}: {m}/{len(rs)} = {100 * m / len(rs):.1f}%")

    disagreements = [r for r in results if not r["match"]]
    if disagreements:
        print("\nDisagreements:")
        for r in disagreements[:10]:
            print(
                f"  {r['scenario_id']}: gold={r['gold']:7} "
                f"predicted={r['predicted_color']:7} ({r['predicted_category']}) | "
                f"amb={r['ambulatory']} br={r['breathing']} "
                f"pulse={r['pulse']} cmd={r['follows_commands']}"
            )

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
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
    plot_exclusion_reasons(cases, os.path.join(out_dir, "syn_starts_exclusion_reasons.png"))