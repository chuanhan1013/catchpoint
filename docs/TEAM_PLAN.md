# CatchPoint — team plan (Thu 22 Oct → Tue 3 Nov 2026)

**Project statement.** CatchPoint is a simulation-validated proposal for a
recoverability-aware safety layer for robotic gait/balance trainers: it preserves the
stroke patient's own reactive step, adds only the minimum pelvic assistance needed, and
escalates to a mechanically analysed catch when recovery is no longer feasible.

**Scope (frozen 22 Oct).**
- **Core claim = I1**: recoverability-based graded control (hands-off → nudge → catch), evidenced by the Monte Carlo study.
- **I2 = feasibility analysis**: walker-mounted cable interface (slack / nudge / slip-limited catch), design D3, "conceptual, unbuilt, unvalidated".
- **Physics demo = supporting demonstration**: pretrained Unitree G1 policy with simulated paretic weakness in MuJoCo, closed-loop B0/B1/CP through the D3 cables (physics_demo/). Shown on one main slide with the Codex-approved wording (calibrated on the same push grid; robot-policy demonstration only). Owner: E; video edit: C.
- **I3 = appendix only**: surrogate-learner study, quoted with its assumptions; never used as evidence for patients or for the core claim.
- No prototype, no clinical-effect claim; every result is labelled "in simulation".

**Briefing items → owner**

| Briefing item (slides 7–8) | Our answer | Owner |
|---|---|---|
| Specific healthcare problem | Falls and missing reactive-step practice during stroke balance training | A |
| Existing solutions and gaps | Threshold-and-lock walkers (MRBA/DRBA), RobUST, mTPAD, overhead BWS | A |
| Specific innovation | I1 control (core); I2 interface (feasibility) | D, B |
| New sensing / interface / mechanism / actuation / control | Sensing fusion (C); cable interface + slip-brake spools (B); control (D) | B, C, D |
| Diagrams, CAD, simulation, video | CAD + load analysis (B); block/timing diagrams (C, D); Monte Carlo (E); demo video (C) | B, C, D, E |
| Close the loop | Limitations + next steps | A |

## Roles (one question each, one talk segment each)

| Role | Question | Definition of done | Slides | Talk |
|---|---|---|---|---|
| **A Clinical — "why"** | What goes wrong for patients today, and how would we know we helped? | Problem slide (cited stats); existing-solutions table with gaps; outcome measures (Berg Balance Scale, fall count, reactive-step metrics); safety/ethics; limitations & next steps; summary; **leads the final claim audit** | 3 (2–4) | ~3 min |
| **B Mechanical — "body"** | Can the walker nudge and catch without tipping? | D3 CAD (cable interface, mixed-height anchors, outriggers, cantilevered BWS anchor, front gate) reconciled with analysis/i2_sizing; FBDs; tension/brake/tip-margin table; allowable arrest envelope | 6 (8–13) | ~3 min |
| **C Sensing — "eyes"** | How does the robot know the patient's state fast enough? | Sensor layout (foot LiDAR, cable-length pelvis tracking, load cells, IMU); XCoM estimation + delay compensation diagram; delay budget; calibration protocol; **records the demo video** from the interactive simulator | 5 (14–18: sensing + physics demo + walking demo) | ~3 min |
| **D Control — "brain"** | When does it act, and how much? | L0/L1/L2 decision flow; minimum-force rule (plain language + appendix equation); timing diagram (decision → force → brake); safety argument; low-level admittance; innovation matrix + three contributions | 3 (5–7) | ~3.5 min |
| **E Simulation — "proof"** | Does it beat today's approach? | Frozen Monte Carlo results (operating points, counts, CIs, ablations CP-pop / CP-noDC / XCoM-only); physics demo; I3 appendix slide (optional, with caveats) | 4 (19–22: results, limitations, validation plan, summary) | ~3.5 min |

**Talk order (one block per presenter; A opens, E closes):** A (2–4) → D (5–7) → B (8–13) → C (14–18) → E (19–22). Four handovers; E closes; every block 3–3.5 min. Concise table: docs/ROLES.md.

**Shared:** one slide template (deck/); each member answers Q&A on their own segment.

## Status on 22 Oct and handover ("take over, verify, complete, present")
Most technical content already exists in the repo. Each member now **takes over** their module: reproduces or reads it, checks
every number they will say against the frozen evidence files, fills the missing pieces below, and presents and defends it.
Role D is taken by Chuanhan (control); he also acts as technical integrator (repo, numbers, final consistency checks), while A
still leads the claim-to-evidence audit.

| Role | Already built (in the repo) | Your job now | Still missing |
|---|---|---|---|
| A Clinical | Problem stats with sources; existing-solutions table; limitations | Read research/*.md; own the problem, gap and limitations slides | Clinical evaluation plan slide (proposed pilot protocol, outcome measures, safety stop rules, ethics) |
| B Mechanical | D3 CAD + front gate; renders; sizing analysis (analysis/i2_sizing, report.md) | Re-run make_report.py; understand FBDs, tip margin, slip brakes, gate load | One slide that explains the load path in plain words; part list / cost estimate |
| C Sensing | Sensor list in SPEC §3; delay values | Own the sensing & estimation slide; record the final demo video | Sensor layout + estimation pipeline + delay budget + calibration protocol figure |
| D Control (Chuanhan) | Controller code (sim/catchpoint), Monte Carlo, physics-demo controller | Own the method slides; lead Q&A on control | Control architecture: decision flow + timing diagram on a main slide |
| E Simulation | Monte Carlo results + CIs; physics demo v1/v2 + PPO; videos | Reproduce run_all.py and one physics video; own the results slides | Short "how to read the results" narration; check every quoted number |

## Interface contracts (agree by Sat 24 Oct)
- **D → B:** nudge ≤ 150 N resultant; catch by passive slip brakes: ≥ 300 N along the fall, ≤ 380 N resultant; arrest load cases from analysis/i2_sizing/data/results.json.
- **B → D/E:** D3 contract in analysis/i2_sizing/data/d3_contract.json (single geometry source for CAD + analysis) and report.md §8 (F_lim, static SF 1.5 envelope, outrigger extension 0.25 m, anchor heights/positions, BWS cantilever anchor, lift cap, walker mass/CG, front gate + latch interlock).
- **C → D:** delay budget: sensing 0.03 s + actuation 0.07 s + lag 0.02 s; catch lock 0.28 s.
- **E → all:** frozen evidence set — results/summary.csv, ci.csv, matched.csv, results/sizing_D3_summary.md, analysis/i2_sizing/data/results.json, results/i3_summary.md, physics_demo/{sweep_all,sweep_pareto}.json, physics_demo/ppo/compare.json. Slides quote only these, with denominators (e.g. "805/814 caught trials").

## Timeline
| Dates | Milestone |
|---|---|
| Thu 22 – Sat 24 Oct | Scope frozen; interface contracts agreed; everyone reads their part of the repo |
| Sun 25 – Wed 28 Oct | Build: D3 CAD + sizing slides (B); diagrams (C, D); results slides (E); problem/gap slides (A) |
| Thu 29 Oct | Draft slides from all five → merged deck |
| Fri 30 – Sat 31 Oct | Claim-to-evidence audit (A leads): no prototype claim, no clinical-effect claim, I2 numbers conditional with denominators, I3 appendix-only; demo video (C) |
| Sun 1 – Mon 2 Nov | Two timed rehearsals (10–15 min) + mock Q&A per role |
| Tue 3 Nov | Submit slides |
