# Novelty check and citable sources - FallSafe proposal (ME5412)

Compiled 2026-10-06. Status tags: [opened] = page or abstract text actually retrieved this session (WebFetch or Europe PMC REST API abstract); [snippet] = only seen in a search-result snippet; [not opened] = fetch blocked (403 / redirect / captcha).
Caveat: WebFetch returns a model summary of the page, not raw text, except where I parsed raw HTML/JSON myself (DRBA HTML, Europe PMC JSON). Numbers taken from WebFetch summaries should be spot-checked against the PDFs before final submission.

## Q1. NOVELTY VERDICT: PARTIALLY NOVEL

No work found that combines all four elements: (1) delay-compensated predicted XCoM / capture point, (2) a patient-specific recoverable region (BoS + reachable recovery step given impaired step length and reaction time), (3) transparency inside the region, (4) minimum graded pelvic force via a control-barrier-function filter with escalation to harness catch/brake. Individual ingredients exist; the combination, and especially the patient-specific step-reachability region plus CBF filter, was not found. No paper found using "control barrier function" for human balance assistance (two targeted searches; absence is a search finding, not proof).

### Classification of close works

1. MRBA, NTU (Ang lab), J NeuroEng Rehabil 2023 [opened, PMC9979429]
   - Does: pelvic-interface mobile robot; force sensors + IMU + potentiometers each compared to fixed thresholds; on detection magnetic brakes lock the arm joints and the base halts. Mechanical response 0.28 s. Self-induced falls: sensitivity 94.7 %, specificity 100 %; FIMP-induced falls: sens 67.5 %, spec 76.3 % (12 healthy young + 1 SCI; second experiment 8 healthy young).
   - Trigger: raw sensor thresholds, no XCoM/CP/MoS. Binary (brake). Does not preserve self-recovery (lock ends the attempt).
   - Vs idea: reactive threshold-and-lock; the baseline the idea improves on.
2. DRBA, NTU (Wang ... Ang), arXiv 2607.03027, 3 Jul 2026 [opened; HTML parsed myself]
   - Does: 3-DoF pelvic arm, "assist-as-needed" in the title; fall detection = sliding-window change in two pelvic force sensors versus adjustable lateral/downward thresholds; on detection the arm brakes lock until the user presses an unlock button. 9 elderly patients; improved step length and speed vs therapist assistance. Four of the six joints have encoders+brakes.
   - Trigger: force-change thresholds, no XCoM/CP. Binary lock despite the "AAN" title. No self-recovery after trigger.
   - Vs idea: same family as MRBA; newest NTU work and the direct incumbent to cite. Also states that end-effector systems like KineAssist compromise transparency and Andago-type systems are bulky.
3. KineAssist (Patton, Brown, Peshkin et al., Top Stroke Rehabil 15(2):131-139, 2008) [snippet only; Academia 403, Northwestern 404]
   - Does: overground pelvis/torso robot, servo-follows, "catches a patient who begins to fall"; clinician-adjustable Safety Zone ("encourage patients to recover from near-falls" per search summary).
   - Binary zone threshold; static, clinician-set, not patient-model-based or predictive.
4. ZeroG / TRiP (Aretech) [opened product page]
   - Overground BWS fall protection "fully customizable for a pre-defined distance or velocity-based response"; TRiP = clinician-set calibrated tugs. Threshold + catch; no CP logic stated. A ZeroG TRiP stroke trial is registered, NCT04919161 [snippet].
5. RobUST pelvic force field, Omofuma et al., Bioengineering 2023 [opened, PMC10740507]  -- CLOSEST CONCEPTUALLY
   - Cable-driven pelvis/trunk robot; restoring force (15 % BW) when the pelvis leaves a subject-specific polygon "stability area" built from the subject's own successful-recovery trials; 10 healthy adults (21-40 y); FF group used larger pelvic excursion with lower MoS; perturbation size progressed by a modified 4-2-1 algorithm.
   - Intervenes only outside the boundary (assist-as-needed, preserves self-recovery inside).
   - Differs: static pelvis-position region, reactive not predictive (no XCoM, no delay compensation, per summary), no step-reachability, no CBF, no escalation, healthy subjects only.
6. Active Pelvis Orthosis, Monaco et al., Sci Rep 7, 2017 [opened, PMC5426188]
   - Wearable hip exo; adaptive oscillators flag deviation (~350 ms to identify); then time-profiled hip torque 0.2 Nm/kg for ~250 ms; 8 elderly (68.9 y) + 2 transfemoral amputees. MoS is an OUTCOME (higher with assist than zero-torque), not the trigger.
   - Graded profile, assist-when-needed. Exo not walker; no recoverable region; no CBF.
7. ExoRecovery, Orhan, Shafiee, Juillard, Oliveira, Ijspeert, Bouri, arXiv 2310.20339, 2023 [opened abstract]
   - Lower-limb exo push recovery; online optimisation of step duration/position from DCM (= XCoM); impedance execution; zero-torque vs recovery-assist modes.
   - Uses XCoM/DCM predictively but the exoskeleton plans and executes the step, i.e. replaces the user's step. Opposite philosophy to preserving the user's own step.
8. Improved/"Projected" XCoM, Sci Rep 2025 (s41598-025-93299-5; PMC11992108) [opened]
   - Velocity and boundary safety coefficients; 92 % vs 84 % accuracy for critical-state classification; triggers ankle vs stepping strategy in a 7-DoF rehab exoskeleton (36.6 Nm hip). Summary says the paper does NOT address delay compensation or step feasibility constraints.
   - Confirms XCoM-as-trigger precedent in exo; no recoverable-region logic.
9. mTPAD / A-TPAD (Agrawal lab), Adeniyi et al., Sci Rep 2023 [opened, PMC10632386]
   - Mobile pelvic robot, ~10 % BW diagonal perturbations, 40 healthy older adults (71 +/- 8 y); BBS higher (p<0.001). XCoM/CP not used. PBT delivery with no safety layer.
10. BAR-TM, Matjacic, Zadravec, Olensek, J NeuroEng Rehabil 2018 [opened, PMC5896154]
    - Treadmill + pelvic push robot, 50-100 N perturbations at foot contact; one stroke subject tolerated 60 N before and 90 N (13 % BW) after training. No XCoM/MoS. Case study.
11. Kumar, Ha, Sawicki, Liu, arXiv 1909.10488 (2019) [opened abstract]
    - Learned fall predictor + recovery policy on an assistive walking device; simulation only.
12. Haarman, Vlutters, Olde Keizer, van Asseldonk, J NeuroEng Rehabil 2017 [opened, EPMC abstract]
    - 10 chronic stroke survivors, mediolateral pelvis perturbations in walking; both legs modulate foot placement; paretic foot placed laterally farther from CoM; more time on the non-paretic leg. Supports leg-specific recoverable regions.
13. Haoyong Yu lab (NUS): found only Aguirre-Ollinger, Narayan, Reyes, Cheng, Yu, ICORR 2019 [opened Europe PMC record: title/authors only]; Mun et al., Med Biol Eng Comput 2017 [snippet]; Aguirre-Ollinger & Yu, J Intell Robot Syst 101:52, 2021 (admittance shaping; only seen in DRBA's reference list; page 404). No NUS paper found using XCoM/CP/MoS triggers or a pelvic safety filter. Coverage of 2018-2026 is NOT exhaustive (Europe PMC returned a single hit).
14. Not classified in depth: Parkinson standing-balance exo (Sensors 2024, 10.3390/s24237498) [snippet]; wearable falls-prevention exo (Sensors 2026, 10.3390/s26144534) [not opened, 403]; Mina v2 (PMC6196256) [not opened, captcha]. Lokomat / FLOAT perturbation features: nothing relevant found.

### Suggested novelty wording
Prior art is (a) threshold-and-lock pelvic catchers (MRBA, DRBA, KineAssist, ZeroG), (b) exoskeletons that use XCoM/DCM or MoS but either replace the step (ExoRecovery) or are single-strategy triggers (PXCoM, APO), (c) a pelvic force field at a subject-specific static boundary (RobUST). Not found: predictive, delay-compensated XCoM vs a patient-specific step-reachability set, with a minimum-norm (CBF/QP) graded pelvic force and harness escalation on a mobile walker. Phrase as "to our knowledge, based on keyword searches of Europe PMC/arXiv/web through Oct 2026", not "first".

## Q2. Numbers and sources

### (a) Falls after stroke
- "Around half of individuals will fall within the 12 months post stroke" is a background statement in Wang et al. 2025 (not their result). Their AVERT cohort (n = 2104, 56 stroke units): 85 (4 %) had >=1 serious fall, 91 falls, 60 % of serious falls caused fracture, 42 % of those hip fracture, 2 deaths. [opened, EPMC] doi 10.1177/17474930251314330. Need a primary source for the ~50 % claim (Batchelor 2012 review is the usual one; its abstract has no numbers).
- Inpatient: "between 14 % and 65 % of people with stroke fall at least once during their hospital admission" (Sullivan 2020 background) [opened, EPMC]. Mansfield 2013: 20 of 136 inpatients (14.7 %) had 29 falls during inpatient rehab [opened, EPMC].
- Case series: 13/21 (62 %) fell within 12 months of outpatient therapy; 41 falls; 28 without appropriate equipment [opened, EPMC] doi 10.1016/j.rh.2023.100809.
- Impaired reactive stepping at discharge: 99/139 (71 %) of inpatient-stroke patients (Inness 2014) [opened, EPMC]. Mansfield 2013: higher inpatient fall rate related to no-step responses, inadequate foot clearance, delayed step initiation, need for external assistance.
- Weerdesteyn 2008, Batchelor 2012: qualitative (falls common in all phases; delayed, less coordinated responses to perturbations) [opened, EPMC].

### (b) Does PBT reduce falls?
- Mansfield 2015 meta-analysis (8 RCTs, n = 404; older adults + neuro populations): faller risk ratio 0.71 (0.52-0.96); fall rate ratio 0.54 (0.34-0.85) [opened, EPMC] doi 10.2522/ptj.20140090.
- Mansfield 2018 chronic-stroke RCT (88 randomised, 83 analysed; 6 wk, 12 sessions): 53 falls (1.45/person-yr) PBT vs 64 (1.72) control, rate ratio 0.85 (0.42-1.69), p = 0.63, NOT significant; per-protocol 32 falls (1.07) vs 57 (1.75) [opened, EPMC] doi 10.1136/bmjopen-2018-021510.
- Brown 2023 systematic review (9 studies, 364 stroke participants): insufficient evidence PBT reduces falls in stroke [opened, EPMC] doi 10.1016/j.clinbiomech.2023.106078.
- Okubo 2017 (stepping training in older adults, 7 RCTs, n = 660): rate ratio 0.48 (0.36-0.65), faller RR 0.51 (0.38-0.68); reactive and volitional similar [opened, EPMC] doi 10.1136/bjsports-2015-095452. Note: stepping training, not PBT specifically.
- Sharma 2025 preprint (older adults, 25 RCTs, n = 2659): rate ratio 0.77 (0.60-0.99), low certainty; programs >= 6 h reduced falls 33 % [opened, EPMC; medRxiv PREPRINT] doi 10.1101/2025.07.23.25331962.
- Handelzalts 2019 subacute stroke RCT (12 x 30 min over 2.5 wk) [snippet only] doi 10.1177/1545968319829453.
- CONTRADICTION: mixed/older-adult meta-analyses show fall reduction; the stroke-specific RCT and review do not reach significance. PBT does improve reactive stepping in stroke (fewer extra steps; Schinkel-Ivy 2019, doi 10.1016/j.jstrokecerebrovasdis.2018.12.011 [opened, EPMC]). Safe claim: plausible mechanism, fall reduction in stroke unproven.

### (c) Reactive stepping timing and step length
- Stroke inpatients at discharge (n = 105): time to foot-off 351 ms paretic vs 365 ms non-paretic (no significant difference); perturbation-evoked falls associated with non-paretic TFO (Inness 2016) [opened, EPMC] doi 10.1097/NPT.0000000000000132.
- Older adults, treadmill walking with lateral perturbations (Nachmani & Melzer 2026; 14 fallers / 21 non-fallers): step initiation 342 +/- 124 vs 306 +/- 122 ms; step duration 670 +/- 195 vs 602 +/- 191 ms; first-step length 21 +/- 10 vs 17 +/- 9 cm (fallers LARGER) [opened, EPMC] doi 10.1123/japa.2025-0192. Walking lateral perturbation, not stance.
- Stance perturbations, stroke (Martinez 2019; 20 stroke, 16 controls): step length shorter than controls (p = 0.001 to 0.025 by leg/stance), step duration longer with the more-involved leg (p = 0.002), more steps with the more-involved leg (p = 0.014) [opened via WebFetch summary] doi 10.3389/fneur.2019.00711.
- Walking: Haarman 2017 paretic foot lands laterally farther from CoM. Step-length asymmetry at discharge in 22/61 (36 %) (Rozanski 2020) [opened, EPMC].
- NOT found: exact paretic vs non-paretic reactive step length in cm; older-adult stance-perturbation foot-off ms. "Response times ~90 ms longer in older adults" appeared only as an unattributed search snippet; do not cite. Maki & McIlroy 2006 abstract has no numbers.

### (d) Canonical definitions
- Hof, Gazendam, Sinke, J Biomech 38(1):1-8, 2005, doi 10.1016/j.jbiomech.2004.03.025 [opened, EPMC abstract]: XcoM = CoM position + velocity x sqrt(l/g); margin of stability b = minimum distance from XcoM to BoS boundary; temporal margin tau.
- Pratt, Carff, Drakunov, Goswami, IEEE-RAS Humanoids 2006, pp. 200-207, doi 10.1109/ICHR.2006.321385 [snippet from search; PDF at ambarish.com/paper/Pratt_Goswami_Humanoids2006.pdf not opened]. Capture region = where the robot must step to stop; LIPM with flywheel.

## Remaining uncertainty
1. Search coverage limited to web search + Europe PMC; no IEEE Xplore/Scopus. A 2025-26 CBF-for-human-balance paper could be missed.
2. Yu lab 2018-2026: only 1-2 papers found; JIRS 2021 page 404.
3. Blocked pages: Springer, Nature, MDPI 2026, Mina v2 PMC, Academia KineAssist. KineAssist Safety Zone is snippet-level.
4. WebFetch outputs are summaries; spot-check MRBA numbers, RobUST 15 % BW, APO 0.2 Nm/kg against PDFs.
5. Missing: cm values for paretic vs non-paretic reactive steps; ms foot-off for older adults on stance perturbation; opened primary source for "~50 % fall in a year".

## References (IEEE style)

[1] MRBA authors not verified, "Mobile Robotic Balance Assistant (MRBA): a gait assistive and fall intervention robot for daily living," J. NeuroEng. Rehabil., 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC9979429/
[2] Y. Wang, L. Li, Y. Wang, C. Yang, S. S. Chan, J. Chen, X. Yan, H. Wang, X. Gong, J. Lin, H. Hu, and W. T. Ang, "DRBA: Dynamic Robotic Balance Assistant - An assist-as-needed gait and balance rehabilitation robot for versatile training," arXiv:2607.03027, Jul. 2026. https://arxiv.org/abs/2607.03027
[3] Omofuma et al., "Training postural balance control with pelvic force field at the boundary of stability," Bioengineering, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10740507/ (title as listed by search; full author list unverified)
[4] Monaco et al., "An ecologically-controlled exoskeleton can improve balance recovery after slippage," Sci. Rep., vol. 7, 2017. https://www.nature.com/articles/srep46721 (read at https://pmc.ncbi.nlm.nih.gov/articles/PMC5426188/)
[5] Z. O. Orhan, M. Shafiee, V. Juillard, J. C. Oliveira, A. Ijspeert, and M. Bouri, "ExoRecovery: Push recovery with a lower-limb exoskeleton based on stepping strategy," arXiv:2310.20339, 2023. https://arxiv.org/abs/2310.20339
[6] "The improved extrapolated center of mass enhances the safety of exoskeleton system," Sci. Rep., 2025. https://www.nature.com/articles/s41598-025-93299-5 (authors not captured)
[7] Adeniyi et al., "Utilizing mobile robotics for pelvic perturbations to improve balance and cognitive performance in older adults: a randomized controlled trial," Sci. Rep., 2023. https://www.nature.com/articles/s41598-023-46145-5
[8] Z. Matjacic, M. Zadravec, and A. Olensek, "Feasibility of robot-based perturbed-balance training during treadmill walking in a high-functioning chronic stroke subject: a case-control study," J. NeuroEng. Rehabil., 2018. https://doi.org/10.1186/s12984-018-0373-z
[9] J. L. Patton, D. A. Brown, M. Peshkin et al., "KineAssist: Design and development of a robotic overground gait and balance therapy device," Top. Stroke Rehabil., vol. 15, no. 2, pp. 131-139, 2008.
[10] V. C. V. Kumar, S. Ha, G. Sawicki, and C. K. Liu, "Learning a control policy for fall prevention on an assistive walking device," arXiv:1909.10488, 2019. https://arxiv.org/abs/1909.10488
[11] J. A. M. Haarman, M. Vlutters, R. A. C. M. Olde Keizer, and E. H. F. van Asseldonk, "Paretic versus non-paretic stepping responses following pelvis perturbations in walking chronic-stage stroke survivors," J. NeuroEng. Rehabil., 2017, doi: 10.1186/s12984-017-0317-z.
[12] G. Aguirre-Ollinger, A. Narayan, F. A. Reyes, H. J. Cheng, and H. Yu, "High mobility control of an omnidirectional platform for gait rehabilitation after stroke," in Proc. IEEE ICORR, 2019, doi: 10.1109/ICORR.2019.8779487.
[13] A. L. Hof, M. G. J. Gazendam, and W. E. Sinke, "The condition for dynamic stability," J. Biomech., vol. 38, no. 1, pp. 1-8, 2005, doi: 10.1016/j.jbiomech.2004.03.025.
[14] J. Pratt, J. Carff, S. Drakunov, and A. Goswami, "Capture point: A step toward humanoid push recovery," in Proc. IEEE-RAS Int. Conf. Humanoid Robots, 2006, pp. 200-207, doi: 10.1109/ICHR.2006.321385.
[15] J. Wang et al., "An exploration of serious falls after stroke using a large international stroke rehabilitation database," Int. J. Stroke, 2025, doi: 10.1177/17474930251314330.
[16] R. Sullivan, K. Harding, I. Skinner, and B. Hemsley, "Falls in hospital patients with acquired communication disability secondary to stroke: A systematic review and meta-analysis," Int. J. Lang. Commun. Disord., 2020, doi: 10.1111/1460-6984.12570.
[17] A. Mansfield, E. L. Inness, J. S. Wong, J. E. Fraser, and W. E. McIlroy, "Is impaired control of reactive stepping related to falls during inpatient stroke rehabilitation?," Neurorehabil. Neural Repair, 2013, doi: 10.1177/1545968313478486.
[18] E. L. Inness, A. Mansfield, B. Lakhani, M. Bayley, and W. E. McIlroy, "Impaired reactive stepping among patients ready for discharge from inpatient stroke rehabilitation," Phys. Ther., 2014, doi: 10.2522/ptj.20130603.
[19] E. L. Inness, A. Mansfield, M. Bayley, and W. E. McIlroy, "Reactive stepping after stroke: Determinants of time to foot off in the paretic and nonparetic limb," J. Neurol. Phys. Ther., 2016, doi: 10.1097/NPT.0000000000000132.
[20] A. Mansfield, J. S. Wong, J. Bryce, S. Knorr, and K. K. Patterson, "Does perturbation-based balance training prevent falls? Systematic review and meta-analysis of preliminary randomized controlled trials," Phys. Ther., 2015, doi: 10.2522/ptj.20140090.
[21] A. Mansfield, A. Aqui, C. J. Danells, S. Knorr, A. Centen, V. DePaul et al., "Does perturbation-based balance training prevent falls among individuals with chronic stroke? A randomised controlled trial," BMJ Open, 2018, doi: 10.1136/bmjopen-2018-021510.
[22] D. Brown, C. Simpkins, and F. Yang, "A systematic review of perturbation-based balance training on reducing fall risk among individuals with stroke," Clin. Biomech., 2023, doi: 10.1016/j.clinbiomech.2023.106078.
[23] Y. Okubo, D. Schoene, and S. R. Lord, "Step training improves reaction time, gait and balance and reduces falls in older people: a systematic review and meta-analysis," Br. J. Sports Med., 2017, doi: 10.1136/bjsports-2015-095452.
[24] S. Sharma et al., "Perturbation-based balance training reduces falls and fall injuries in older people," medRxiv preprint, 2025, doi: 10.1101/2025.07.23.25331962.
[25] H. Nachmani and I. Melzer, "Differences in the first reactive step response to unannounced lateral perturbation during walking in older adults with and without a fall history," J. Aging Phys. Act., 2026, doi: 10.1123/japa.2025-0192.
[26] K. M. Martinez, M. W. Rogers, M. T. Blackinton, M. S. Cheng, and M.-L. Mille, "Perturbation-induced stepping post-stroke: A pilot study demonstrating altered strategies of both legs," Front. Neurol., vol. 10, 2019, doi: 10.3389/fneur.2019.00711.
[27] B. E. Maki and W. E. McIlroy, "Control of rapid limb movements for balance recovery: age-related changes and implications for fall prevention," Age Ageing, 2006, doi: 10.1093/ageing/afl078.
[28] F. A. Batchelor, S. F. Mackintosh, C. M. Said, and K. D. Hill, "Falls after stroke," Int. J. Stroke, 2012, doi: 10.1111/j.1747-4949.2012.00796.x.
[29] G. M. Rozanski et al., "Longitudinal change in spatiotemporal gait symmetry after discharge from inpatient stroke rehabilitation," Disabil. Rehabil., 2020, doi: 10.1080/09638288.2018.1508508.

[30] World Health Organization, "Falls — fact sheet," 26 Apr. 2021. https://www.who.int/news-room/fact-sheets/detail/falls (opened 2026-10-06): "Each year an estimated 684 000 individuals die from falls globally"; 37.3 million falls per year are severe enough to require medical attention.
