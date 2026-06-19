# KG-DQN: A Knowledge-Graph-Constrained Reinforcement Learning Agent for Mass Casualty Triage

This repository contains the code, ontology, trained checkpoints, and evaluation results for a short paper submitted to **KG-NeSy 2026** (Third International Workshop on Knowledge Graphs and Neurosymbolic AI, co-located with ISWC 2026, Bari, Italy).

> **In one sentence:** we encode the START mass-casualty triage protocol (plus CBRN decontamination rules) as a reasoner-validated OWL/SWRL ontology, and use it to *structurally* block a Deep Q-Network agent from ever taking a clinically unsafe action — instead of just penalizing unsafe actions and hoping the agent learns to avoid them.

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [The core idea, explained simply](#the-core-idea-explained-simply)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [How to run everything, in order](#how-to-run-everything-in-order)
- [File-by-file guide: what each file does and what its output means](#file-by-file-guide-what-each-file-does-and-what-its-output-means)
- [Results folder](#results-folder)
- [Citing this work](#citing-this-work)

---

## Why this project exists

In a mass casualty incident (MCI) — a disaster, attack, or accident with more patients than available medical resources — first responders use the **START protocol** to quickly sort patients into four categories: **Minor**, **Delayed**, **Immediate**, or **Expectant**. When the incident also involves chemical, biological, radiological, or nuclear (CBRN) contamination, patients additionally need **decontamination before treatment**.

A reinforcement learning (RL) agent can, in principle, learn this sorting task from simulated experience. But a standard ("vanilla") RL agent has **no guarantee** it won't take a clinically wrong action — especially for rare patient types it barely saw during training. In a domain like triage, a wrong action means misallocating scarce resources away from a patient who could have been saved. That's not an acceptable failure mode, no matter how rare.

This project builds a **symbolic safety layer**: an OWL/RDF ontology that encodes the START rules and CBRN rules as formal logic, validated by an actual automated reasoner (not just hand-written if/else code), which is then used to **mask out** (make unselectable) any action the agent could take that would violate the protocol. The RL agent literally cannot pick a forbidden action — it's not that it's been trained to avoid it, it *physically isn't an option*.

## The core idea, explained simply

Think of it like this: a vanilla RL agent is a student who has only ever been *graded down* for wrong answers — given enough practice, they usually learn to avoid mistakes, but nothing stops them from occasionally making one, especially on questions they rarely see. Our agent is a student taking a multiple-choice test where the genuinely wrong answers have been **physically removed from the page** for certain rare/critical scenarios. They can still choose poorly among what's left, but they can never select an answer that was never offered.

We compare two agents trained on identical patient scenarios:
- **Vanilla DQN** — a standard Deep Q-Network. Wrong/unsafe actions are heavily penalized (–50 reward) but still *executable*.
- **KG-DQN** — the same network architecture, but unsafe actions are removed from its choices entirely via the ontology-derived mask.

We then test both **exhaustively** — not just on random episodes, but on *every one of the 80 possible patient profiles* the environment can generate — across three different random training seeds, to see whether the safety guarantee actually holds everywhere, and whether it costs anything in accuracy.

---

## Repository structure

```
.
├── ontology/
│   └── triage_v2.rdf              # The knowledge graph itself (OWL/RDF + SWRL rules)
├── src/
│   ├── env/
│   │   ├── dqn.py                 # Neural network architecture + replay buffer
│   │   ├── mci_env.py             # The Gymnasium RL environment (the "world" the agent acts in)
│   │   └── patient_generator.py   # Generates random simulated patients
│   ├── kg/
│   │   └── constraint_guard.py    # Bridge between the ontology and Python; runs the reasoner
│   └── eval/
│       └── exhaustive_eval.py     # Defines and runs the 80-profile exhaustive test
├── train/
│   ├── train_dqn.py               # The training loop for both agents
│   └── random_agent.py            # Sanity-check baseline (acts completely randomly)
├── tests/
│   ├── tests_kg.py                # Unit tests for the ontology/reasoner logic alone
│   ├── tests_hazard.py            # Unit tests for CBRN hazard/decontamination logic
│   ├── tests_env.py               # Tests the Gymnasium environment wiring
│   ├── tests_dqn.py               # Quick single-seed training + comparison (dev smoke test)
│   └── tests_robustness.py        # THE paper's main experiment: multi-seed, exhaustive evaluation
├── results/                       # All generated outputs land here (created automatically)
│   ├── *.pt                       # Trained model checkpoints
│   ├── *.csv                      # Per-profile evaluation results
│   └── *.png                      # Figures
├── paper/
│   ├── main.tex                   # The KG-NeSy 2026 short paper
│   └── references.bib             # Bibliography
└── README.md                      # This file
```

---

## Setup

```bash
conda create -n ontologies python=3.10
conda activate ontologies
pip install gymnasium numpy torch owlready2 matplotlib
```

> **Note on the reasoner:** `constraint_guard.py` uses Pellet via `owlready2`. Pellet requires a Java runtime (JRE/JDK) to be installed and on your `PATH`, since Pellet itself is a Java program that `owlready2` calls out to.

All scripts assume they are run from the **repository root** (e.g., `python tests/tests_robustness.py`, not from inside the `tests/` folder), since several files resolve paths like `ontology/triage_v2.rdf` relative to the working directory.

---

## How to run everything, in order

If you're setting this up fresh, run things in this order — each step builds on the last:

1. **`python tests/tests_kg.py`** — confirms the ontology and reasoner are working correctly, in isolation, before anything else touches them.
2. **`python tests/tests_hazard.py`** — confirms the CBRN decontamination logic is wired correctly.
3. **`python tests/tests_env.py`** — confirms the RL environment itself behaves correctly (action masking, rewards, multi-step decontamination).
4. **`python tests/tests_dqn.py`** *(optional, quick)* — a fast single-seed sanity check that training runs end-to-end without crashing, and produces a first comparison plot.
5. **`python tests/tests_robustness.py`** — **this is the real experiment.** Trains both agents three times (three random seeds), evaluates them exhaustively, and produces every number and figure reported in the paper. This takes a while — it trains four full models (Vanilla + KG-DQN, × 3 seeds, plus the budget-ablation re-evaluation).

Steps 1–3 are fast (seconds to low minutes). Step 5 is the long one — budget real time for it.

---

## File-by-file guide: what each file does and what its output means

### `ontology/triage_v2.rdf`

This is **not Python code** — it's the knowledge graph itself, written in OWL/RDF (a formal logic format), editable visually in a tool called Protégé. It defines:
- The four START categories (**Minor**, **Delayed**, **Immediate**, **Expectant**) as logical definitions over five patient properties (ambulation, breathing, pulse, mental status, decontamination status) — not hardcoded rules, but formal class definitions a reasoner can derive conclusions from.
- Four CBRN hazard types (Chemical, Biological, Radiological, Nuclear), each with a required number of decontamination steps.
- SWRL rules that *independently* compute a recommended treatment action and severity score, used to cross-check the formal class definitions agree with each other.

You won't run this file directly — every other file in `src/kg/` and `src/env/` reads it.

---

### `src/kg/constraint_guard.py`

This is the bridge between the ontology file and your Python code. On load, it:
1. Reads `triage_v2.rdf`.
2. Runs the **Pellet reasoner** once, over all 32 possible binary patient profiles, computing both the formal-logic classification *and* the SWRL-rule-based classification, and checking they agree (if they don't, it raises an error immediately rather than silently using bad data).
3. Caches the result into a fast lookup table, so every later check during training/evaluation is just a dictionary lookup rather than re-running the reasoner each time.

Exposes functions like `check_action()` (is this action allowed for this patient?) and `get_valid_actions()` (the full mask of allowed actions), used by the environment on every single step.

---

### `src/env/patient_generator.py`

Generates one random simulated patient: random ambulatory/breathing/pulse/mental-status/decontamination status, plus a random CBRN hazard if contaminated, plus the "correct" ground-truth triage tag computed from the same START logic (with 15% of patients having their label randomly escalated, to simulate real-world clinician override/uncertainty).

---

### `src/env/mci_env.py`

The actual **Gymnasium environment** — the simulated "world." Each episode presents 10 patients in sequence. The agent has 7 possible actions per step (4 tagging actions that end a patient's turn, plus `open_airway`, `treat`, `decontaminate` which keep the same patient active for up to 4 actions). If `use_kg_constraint=True`, the action mask from `constraint_guard.py` is enforced — the agent cannot select a blocked action at all. If `False` (Vanilla mode), a blocked action is allowed but penalized –50.

---

### `src/env/dqn.py`

The neural network itself: a small feed-forward network (11 inputs → 64 → 64 → 7 outputs) that estimates how good each of the 7 actions is in a given patient state, plus a replay buffer (a memory of past experience used to stabilize training, standard in DQN).

---

### `src/eval/exhaustive_eval.py`

Defines the **80 enumerable patient profiles** (16 combinations of the 4 binary vital signs, × 5 contamination states: clean or one of 4 hazards) and runs a trained model against every single one, deterministically, recording whether it got the correct tag and whether it ever violated the ontology along the way. This is the rigorous evaluation — not a random sample, the *entire* space of possible initial patients.

---

### `train/train_dqn.py`

The training loop. Standard epsilon-greedy DQN with a target network. The one important design choice: every 20 episodes, it checks the current network against the full 80-profile exhaustive set (not just a rolling average of recent training episodes) and only saves the checkpoint if that score improved. This prevents accidentally keeping a lucky-but-bad snapshot just because training happened to end at the right/wrong moment.

---

### `tests/tests_kg.py`

**Run:** `python tests/tests_kg.py`

Eight isolated unit tests confirming the ontology + reasoner correctly encode each START rule and the CBRN decontamination rule, with no RL agent involved at all — pure symbolic logic checks.

**📸 Terminal output of `tests_kg.py`**

![Terminal output of tests_kg.py](terminal_outputs/tests_kg.png)
**What the output means:** each line printed is `PASS scenario N: <description>`, ending in `All 8 scenarios passed.` Every `PASS` confirms one specific clinical rule is being enforced correctly by the reasoner — for example, "ambulatory patient" confirms a walking patient can only be tagged Minor, and "CBRN Rule 5" confirms `treat` is blocked on a contaminated patient until decontamination completes. If any line says `FAIL` instead of `PASS`, or the script crashes with an `AssertionError`, it means the ontology itself has a logical inconsistency — this must be fixed before trusting any later result, since every other file in this repo depends on this file's correctness.

---

### `tests/tests_hazard.py`

**Run:** `python tests/tests_hazard.py`

Confirms the CBRN hazard/decontamination-duration wiring specifically: that nuclear hazard requires 3 decontamination steps, chemical requires 1, that `treat` stays blocked until decontamination finishes and becomes valid immediately after, and that a patient with no assigned hazard type correctly falls back to a default duration.

**📸 Terminal output of `tests_hazard.py`**

![Terminal output of tests_hazard.py](terminal_outputs/tests_hazard.png)
**What the output means:** each printed line states an expected value and the actual computed value side by side (e.g., `Nuclear hazard decon duration: 3 (expect 3)`), followed by `All hazard-type wiring checks passed.` This confirms the multi-step decontamination mechanic — the part of this project that makes the CBRN extension genuinely sequential rather than a single static lookup — is working correctly end to end.

---

### `tests/tests_env.py`

**Run:** `python tests/tests_env.py`

Tests the Gymnasium environment itself: correct observation shape, correct action-space size, that the KG-constrained mode actually blocks wrong actions (re-confirming `tests_kg.py`'s findings, but now through the environment rather than the ontology layer directly), and a full multi-step decontaminate-then-treat sequence.

**📸 Terminal output of `tests_env.py`**
![Terminal output of tests_env.py](terminal_outputs/tests_env.png)

**What the output means:** look for `Episode done: N raw env.step() calls for 10 patients` — this confirms a full 10-patient episode completes correctly. The `Valid action mask: [...]` line shows the actual boolean array of which of the 7 actions are currently allowed for one patient. `Ambulatory patient tagged Immediate → reward: -50.0` confirms that, exactly as expected, the KG penalty fires when the environment is asked to do something the ontology forbids. The decontamination-flow section at the end shows `decontaminated` flipping from `False` to `True` exactly when the required number of decontaminate actions accumulates, and `progress` climbing from 0.0 to 1.0 — confirming the multi-step CBRN mechanic works correctly inside the actual environment, not just in isolated logic.

---

### `tests/tests_dqn.py`

**Run:** `python tests/tests_dqn.py`

A **quick, single-seed** end-to-end smoke test: trains both a random agent and both DQN variants once, prints a summary comparison table, and saves one comparison figure. This is meant for fast iteration during development — not the numbers reported in the paper.

**📸 Terminal output of `tests_dqn.py`, the final summary table + convergence section**
![Terminal output of tests_env.py](terminal_outputs/tests_env.png)

**What the output means:** the summary table compares four agents (Random, Random+KG, Vanilla DQN, KG-DQN) across Reward, Correct%, and Violations. The key thing to look for: **KG-DQN's Violations column should always read exactly 0**, regardless of training quality — this is the masking guarantee, visible from the very first run. The "Convergence to 80% correct tagging" section reports which training episode each agent first crossed 80% accuracy — useful for the training-dynamics figure, but not the paper's headline safety/accuracy numbers, which come from `tests_robustness.py` instead.

**📸 [SCREENSHOT 5 GOES HERE — the `results/full_comparison.png` figure this script generates]**

**What this figure means:** four panels, left to right — episode reward over training, correct-tag rate over training, KG violations per episode (Vanilla descends slowly but never reaches zero; KG-DQN is flat at zero from the start), and the gap between raw accuracy and KG-agreement rate (reflecting the 15% noisy-label patients in the training data).

---

### `tests/tests_robustness.py`

**Run:** `python tests/tests_robustness.py`

**This is the main event — the actual experiment reported in the paper.** For each of three random seeds (42, 43, 44):
1. Trains a fresh Vanilla DQN and a fresh KG-DQN from scratch (500 episodes each).
2. Evaluates both exhaustively against all 80 patient profiles.
3. Confirms (via an `assert`) that KG-DQN's violation rate is exactly zero.

After all three seeds, it reports the **multi-seed summary** (mean ± standard deviation accuracy and violation rate for each agent — the headline numbers in Table 1 of the paper), then does a **deep dive on the first seed only**: saves per-profile CSVs, generates a rarity-vs-correctness figure, lists every failing patient profile for both agents with full step-by-step action traces, and finally re-evaluates the same trained KG-DQN checkpoint with a larger action budget (8 instead of 4) — with no retraining — to check whether its remaining failures are a budget artifact or a genuine gap.

This takes the longest to run of anything in this repo, since it trains 6 full models total (2 agents × 3 seeds) plus the extra budget re-evaluation.

**📸 [SCREENSHOT 6 GOES HERE — terminal output: the three "SEED" training blocks]**

**What this output means:** for each seed, you'll see two long blocks of `EP N | Reward: ... | Correct: ... | KG-agree: ... | Violations: ... | Eps: ... | Exhaustive-acc: ... (viol ...)` lines — one block for Vanilla, one for KG-DQN. The `Exhaustive-acc` and `(viol ...)` numbers are the trustworthy ones (checked against all 80 profiles every 20 episodes); the `Correct`/`Violations` numbers earlier in the line are noisier, since they're from just that one 10-patient episode. **Critically, KG-DQN's `Violations` column reads `0` on literally every single line, from episode 0 onward** — this is not something it learned, it's a structural guarantee of the masking. Each seed block ends with `Saved BEST checkpoint (exhaustive 80-profile accuracy X.XX) to results/...pt` — confirming the script kept the best-performing snapshot across all 26 checkpoints it evaluated, not just whatever existed when training happened to stop.

**📸 [SCREENSHOT 7 GOES HERE — terminal output: the "MULTI-SEED SUMMARY" table]**

**What this output means:** this table is the paper's Table 1. It reports, across the three seeds, mean ± standard deviation accuracy and violation rate for both agents. **KG-DQN's violation rate should read `0.0% ± 0.0%`** — zero average, zero spread, because it's mathematically guaranteed rather than empirically observed. Vanilla's numbers will have real variance — this is expected and is itself part of the point: an unconstrained agent's safety is incidental to how training happened to go, not guaranteed.

**📸 [SCREENSHOT 8 GOES HERE — terminal output: the "Vanilla DQN failure cases" and "KG-DQN failure cases" listings]**

**What this output means:** this is the failure-mode analysis (Section 4.2 of the paper). Each line lists one specific patient profile that an agent got wrong or violated the KG on, with its prior probability (how common that exact patient type is in the real-world training distribution — note how tiny these numbers are, e.g. `p=0.0014`, confirming these are genuinely the rarest cases). For Vanilla, look at the `correct_tag` vs `final_tag` columns — a mismatch here is a genuine wrong-answer mistake. For KG-DQN, look for `final_tag=None` — this means the agent ran out of its 4-action budget without ever tagging the patient (a "timeout"), which is a structurally safer failure than Vanilla's wrong-tag failures, since no forbidden action was ever taken.

**📸 [SCREENSHOT 9 GOES HERE — terminal output: the full action-by-action traces for KG-DQN's failures]**

**What this output means:** for each KG-DQN failure, you get a literal step-by-step replay: which action was taken, what reward it received, and whether it was KG-valid. This is your explainability evidence — instead of saying "the agent stalled," you can point to the exact sequence (e.g., repeatedly selecting `open_airway` after it already took effect once, paying a redundancy penalty, then running out of budget) and explain precisely why it failed.

**📸 [SCREENSHOT 10 GOES HERE — terminal output: the "Budget-sensitivity check" section]**

**What this output means:** this re-tests the *same already-trained* KG-DQN model (no retraining at all) with double the action budget, to check if more "thinking time" alone would have fixed the failures above. The conclusion line at the end tells you the verdict directly: `0/N previously-failing profiles became correct purely from a larger action budget` means the problem is a genuine gap in what the model learned, not just an unfairly tight time limit.

**📸 [SCREENSHOT 11 GOES HERE — the `results/robustness_stress_test.png` figure]**

**What this figure means:** two panels. Left: correctness as a function of how rare a patient profile is (rarest on the left) — both agents tend to dip on the rarest profiles. Right: violation rate as a function of rarity — **KG-DQN's line should be flat at zero across the entire plot**, while Vanilla's line rises specifically in the rare region. This is the clearest single visual proof that the safety guarantee holds everywhere, not just on average.

---

### `train/random_agent.py`

**Run:** `python train/random_agent.py`

A pure sanity-check baseline: an agent that picks actions completely at random, with and without KG masking, over 50 episodes each. Useful as a "floor" reference — if a trained agent doesn't clearly beat this, something is wrong with training.

---

## Results folder

After running `tests_robustness.py`, your `results/` folder will contain:

| File | What it is |
|---|---|
| `vanilla_dqn_seed{42,43,44}.pt` / `kg_dqn_seed{42,43,44}.pt` | The best trained model weights for each agent, each seed |
| `vanilla_dqn.pt` / `kg_dqn.pt` | Copies of the seed-42 ("primary seed") checkpoints, for any script that expects a fixed filename |
| `robustness_vanilla.csv` / `robustness_kg.csv` | Per-profile results (all 80 rows) for the primary seed — every profile's correct tag, predicted tag, violation flag, etc. |
| `robustness_kg_budget8.csv` | Same, but for the budget-sensitivity re-evaluation |
| `robustness_stress_test.png` | The rarity-vs-correctness/violation figure (Figure 1 in the paper) |
| `full_comparison.png` | The training-dynamics figure from `tests_dqn.py` (Figure 2 in the paper) |

These CSVs and PNGs are what's referenced in the paper's acknowledgments section as supporting evidence for every reported number.

---

## Citing this work

If you use this code or ontology, please cite:

```bibtex
@inproceedings{fatima2026kgdqn,
  author    = {Fatima, Maham and Tolg, Boris},
  title     = {A Triage Ontology-Guided Knowledge Graph for Constraining
               Reinforcement Learning Decisions in Mass Casualty Incidents},
  booktitle = {Proceedings of the 3rd International Workshop on Knowledge
               Graphs and Neurosymbolic AI (KG-NeSy 2026), co-located with
               ISWC 2026},
  year      = {2026},
  address   = {Bari, Italy}
}
```

---

## Acknowledgments

Developed as part of doctoral research at HAW Hamburg, Faculty of Life Sciences, in collaboration with ongoing work on AI-supported maritime CBRN emergency response.
