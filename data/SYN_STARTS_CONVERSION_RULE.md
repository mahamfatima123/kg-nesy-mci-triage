# Locked Conversion Rule — Syn-STARTS Structured Vitals → Ontology Features

Applied mechanically to the `vitals_info` JSON object. No prose is read;
no judgment call is made. A case is included only if all four
conversions below succeed; otherwise it is excluded, not guessed.

## ambulatory
`vitals_info.can_walk` (boolean) → used directly as `ambulatory`.

## breathing
- If `respirations.rate` is present: `breathing = (rate > 0)`.
- Else if `respirations.breathing_after_maneuver` is present (used for
  cases where initial breathing was absent and airway-opening was
  attempted): `breathing = respirations.breathing_after_maneuver`.
  This matches START's own decision logic exactly: the
  breathing-after-maneuver value IS the protocol-relevant breathing
  status once airway opening has been attempted.
- Else: not derivable, case excluded.

## pulse
`vitals_info.perfusion.radial_pulse_present` (boolean) → used directly
as `pulse`. If `perfusion` key or `radial_pulse_present` subfield is
absent, not derivable, case excluded.

## follows_commands
`vitals_info.mental_status.obeys_commands` (boolean) → used directly as
`follows_commands`. If `mental_status` key or `obeys_commands`
subfield is absent, not derivable, case excluded.

## Note on tag casing
The corpus contains both "Black" and "BLACK" as tag values for the
Expectant category (a generation artifact); both are treated as the
same gold label.