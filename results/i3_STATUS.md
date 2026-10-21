# I3 status — appendix only (frozen 22 Oct 2026)

**How to quote I3 (exact wording):**
> I3 is a four-parameter surrogate-learning study, not patient evidence. With one hand-tuned reward,
> 12 paired seeds per severity, 240 unique training pushes (1,200 candidate evaluations per learner) and
> 0.2–0.6 m/s disturbances, CatchPoint produced 2–5 training falls versus 90–121 for no robot, and improved
> no-robot retention (SELF) for mild (+19.7 ± 9.3 pp vs no robot) and moderate (+7.2 ± 6.1 pp) surrogate
> learners. The effect disappeared or reversed under the error-only reward (λ = 0), and severe learners did not
> improve. These results are properties of this surrogate, reward and disturbance dose; they do not establish
> patient learning or clinical benefit.

**Known reasons (from Codex Round-2 review, review/codex_round2.md §2):**
- λ = 0 degeneracy: reward becomes exactly +1/−1; under B1/B2 nearly all candidates tie at +1, so the rank-based
  ES makes no update. This is a reward degeneracy, not evidence of dependence or of its absence.
- Dose: I3 uses 0.2–0.6 m/s (the 0.3–0.8 m/s range gave a learning plateau) while the main study uses
  0.2–0.9 m/s (SPEC §5.2) — a selection/sensitivity caveat.
- Severe learners stay at ≈48 % retention (untrained level): physical limits dominate the learnable skill.

**Use:** one appendix slide at most, with this wording; never cited in the core claim.
Data: results/i3_summary.md, results/i3_results.csv, results/i3_paired_vs_B0.csv, results/figs/i3_*.png.
