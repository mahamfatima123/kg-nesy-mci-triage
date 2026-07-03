"""
Locked conversion rule: Syn-STARTS structured vitals -> ontology
features. See data/SYN_STARTS_CONVERSION_RULE.md for the full,
human-readable specification this file implements mechanically.
"""

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
    """breathing = (rate > 0) if a rate is given; else
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


def _extract_pulse(perfusion):
    """Reads pulse status from either radial_pulse_present directly,
    or capillary_refill_seconds <= 2 as the clinically equivalent
    perfusion check (both are valid START perfusion criteria)."""
    if perfusion is None:
        return None
    if "radial_pulse_present" in perfusion and isinstance(
        perfusion["radial_pulse_present"], bool
    ):
        return perfusion["radial_pulse_present"]
    if "capillary_refill_seconds" in perfusion and isinstance(
        perfusion["capillary_refill_seconds"], (int, float)
    ):
        return perfusion["capillary_refill_seconds"] <= 2
    return None


def convert_case(case):
    """Returns (ambulatory, breathing, pulse, follows_commands,
    respiratory_rate) or None if any of the four cannot be
    mechanically derived under the locked conversion rule."""
    v = case["vitals_info"]

    if "can_walk" not in v or not isinstance(v["can_walk"], bool):
        return None
    ambulatory = v["can_walk"]

    respirations = v.get("respirations")
    breathing = _extract_breathing(respirations)
    if breathing is None:
        return None

    respiratory_rate = None
    if respirations is not None and "rate" in respirations and isinstance(
        respirations["rate"], (int, float)
    ):
        respiratory_rate = int(respirations["rate"])

    perfusion = v.get("perfusion")
    pulse = _extract_pulse(perfusion)
    if pulse is None:
        return None

    mental_status = v.get("mental_status")
    if mental_status is None or "obeys_commands" not in mental_status or not isinstance(
        mental_status["obeys_commands"], bool
    ):
        return None
    follows_commands = mental_status["obeys_commands"]

    return ambulatory, breathing, pulse, follows_commands, respiratory_rate