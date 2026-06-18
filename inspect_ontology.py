"""
inspect_ontology.py
Run from your project root:  python inspect_ontology.py

A complete, accurate dump of triage_v2.rdf -- classes, BOTH asserted and
reasoner-inferred hierarchy, every property (object + data) with domain/
range, every EquivalentTo definition shown as the literal expression (not
a placeholder), every disjointness axiom including n-ary ones, every
individual with ALL of its property assertions, the SWRL rules, and a
live demo of the reasoner actually classifying a few test patients.

This replaces extract_ontology.py, which had three blind spots:
  1. It only showed the ASSERTED class hierarchy (no reasoner run), so
     Minor/Delayed/Immediate/Expectant looked like orphaned top-level
     classes instead of Patient subclasses -- they ARE Patient subclasses
     once you reason over the equivalentClass definitions, you just
     couldn't see that without running HermiT.
  2. Its disjointness loop only printed axioms with exactly 2 members
     (`if len(pair) == 2`), silently skipping the 4-way
     AllDisjoint(Minor, Delayed, Immediate, Expectant) axiom.
  3. It never printed object/data property VALUES on individuals (e.g.
     requiresAction on the priority individuals), only class membership.
"""

from owlready2 import get_ontology, sync_reasoner, Thing
import os

OWL_PATH = "ontology/triage_v2.rdf"


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


onto = get_ontology(f"file://{os.path.abspath(OWL_PATH)}").load()
print(f"\nOntology loaded: {onto.base_iri}")
print(f"Classes: {len(list(onto.classes()))}")
print(f"Individuals: {len(list(onto.individuals()))}")
print(f"Object properties: {len(list(onto.object_properties()))}")
print(f"Data properties: {len(list(onto.data_properties()))}")

all_classes = list(onto.classes())
all_object_properties = list(onto.object_properties())
all_data_properties = list(onto.data_properties())

# ── 1. Asserted class hierarchy ─────────────────────────────────────────
section("ASSERTED CLASS HIERARCHY  (before reasoning -- raw rdfs:subClassOf)")
top_level = [c for c in all_classes
             if all((not hasattr(p, "name")) or (p.name == "Thing") for p in c.is_a)]
for c in top_level:
    comment = c.comment[0][:65] if c.comment else ""
    print(f"\n  [{c.name}]  {comment}")
    for sub in c.subclasses():
        equiv_str = "  [EquivalentTo defined]" if sub.equivalent_to else ""
        print(f"    +-- {sub.name}{equiv_str}")

# ── 2. EquivalentTo definitions -- literal expressions, not placeholders ─
section("EQUIVALENTTO DEFINITIONS  (the literal OWL-DL restriction)")
for c in all_classes:
    if c.equivalent_to:
        print(f"\n  {c.name} \u2261")
        for expr in c.equivalent_to:
            print(f"    {expr}")
        if c.comment:
            print(f"    // {c.comment[0]}")

# ── 3. Disjointness axioms, including n-ary ones ─────────────────────────
section("DISJOINTNESS AXIOMS  (all group sizes, via onto.disjoint_classes())")
for d in onto.disjoint_classes():
    names = [e.name for e in d.entities if hasattr(e, "name")]
    print(f"  AllDisjoint({', '.join(names)})")

# ── 4. Object properties ─────────────────────────────────────────────────
section("OBJECT PROPERTIES  (domain -> property -> range)")
for p in all_object_properties:
    dom = [d.name for d in p.domain if hasattr(d, "name")]
    ran = [r.name for r in p.range if hasattr(r, "name")]
    sym = " [symmetric]" if "Symmetric" in [b.__name__ for b in p.is_a if hasattr(b, "__name__")] else ""
    print(f"  {dom[0] if dom else '?':<16} --[{p.name}]--> {ran[0] if ran else '?'}{sym}")

# ── 5. Data properties ───────────────────────────────────────────────────
section("DATA PROPERTIES")
for p in all_data_properties:
    dom = [d.name for d in p.domain if hasattr(d, "name")]
    print(f"  {dom[0] if dom else '?':<16} --[{p.name}]--> literal")

# ── 6. SWRL rules ─────────────────────────────────────────────────────────
section("SWRL RULES  (declarative spec -- NOT executed by HermiT)")
for rule in onto.rules():
    label = rule.label[0] if rule.label else "(no label)"
    print(f"\n  {label}")
    print(f"    {rule}")

# ── 7. Individuals with EVERY property assertion ──────────────────────────
section("NAMED INDIVIDUALS  (full property assertions, not just type)")
groups = {}
for ind in onto.individuals():
    asserted_types = [t.name for t in ind.is_a if hasattr(t, "name")]
    key = asserted_types[0] if asserted_types else "Unknown"
    groups.setdefault(key, []).append(ind)

for cls_name in sorted(groups.keys()):
    print(f"\n  {cls_name}:")
    for ind in groups[cls_name]:
        print(f"    * {ind.name}")
        if hasattr(ind, "comment") and ind.comment:
            print(f"        comment : {ind.comment[0][:80]}")
        for p in all_object_properties:
            vals = list(getattr(ind, p.python_name, []) or [])
            if vals:
                print(f"        {p.name:<22} -> {[v.name for v in vals]}")
        for p in all_data_properties:
            vals = list(getattr(ind, p.python_name, []) or [])
            if vals:
                print(f"        {p.name:<22} -> {vals}")

# ── 8. Run the reasoner, then show INFERRED hierarchy ──────────────────────
section("RUNNING REASONER (HermiT)")
with onto:
    sync_reasoner(infer_property_values=True, debug=0)
print("Reasoner finished.")

section("INFERRED CLASS HIERARCHY  (after reasoning -- entailed subsumption)")
patient_subclasses = [c for c in all_classes if Thing in c.ancestors() and
                       onto.Patient in c.ancestors() and c is not onto.Patient]
print(f"\n  [Patient]")
for sub in sorted(set(patient_subclasses), key=lambda c: c.name):
    print(f"    +-- {sub.name}  (now correctly shown under Patient post-reasoning)")

# ── 9. Live classification demo ────────────────────────────────────────────
section("LIVE CLASSIFICATION DEMO  (reasoner classifying fresh test patients)")
demo_profiles = [
    ("demo_ambulatory",      True,  True,  True,  True,  True),
    ("demo_stable",          False, True,  True,  True,  True),
    ("demo_no_commands",     False, True,  True,  False, True),
    ("demo_no_pulse",        False, True,  False, True,  True),
    ("demo_dead",            False, False, False, True,  True),
    ("demo_contaminated",    False, True,  True,  True,  False),
]
with onto:
    demo_inds = {}
    for name, amb, br, pulse, cmd, dec in demo_profiles:
        p = onto.Patient(name)
        p.hasAmbulationStatus = [onto.ambulatory if amb else onto.nonAmbulatory]
        p.hasRespiratoryStatus = [onto.breathing if br else onto.notBreathing]
        p.hasPulseStatus = [onto.pulsePresent if pulse else onto.pulseAbsent]
        p.hasMentalStatus = [onto.followsCommands if cmd else onto.doesNotFollowCommands]
        p.hasDecontaminationStatus = [onto.decontaminated if dec else onto.notDecontaminated]
        demo_inds[name] = p
    sync_reasoner(infer_property_values=True, debug=0)

tag_classes = [onto.Minor, onto.Delayed, onto.Immediate, onto.Expectant]
for name, p in demo_inds.items():
    inferred = [c.name for c in tag_classes if c in p.INDIRECT_is_a]
    decon_flag = onto.RequiresDecontaminationFirst in p.INDIRECT_is_a
    print(f"  {name:<20} -> classified as: {inferred[0] if inferred else '???'}"
          f"{'  [requires decon first]' if decon_flag else ''}")

print("\n" + "=" * 70)
print("  Inspection complete.")
print("=" * 70)
