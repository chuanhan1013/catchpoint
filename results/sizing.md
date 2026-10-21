# Actuator sizing (from CP Monte Carlo, default m = 0.03, 2000 trials, seed 5412)

Per-trial peaks are taken over each trial (while the patient has not fallen); percentiles are across trials.
"All" = all 2000 trials; "L1-active" = the 1344 trials in which the pelvic force exceeded 0.5 N.
The pelvic SEA force excludes the harness catch force.

| Quantity | All trials | L1-active trials |
|---|---|---|
| 99th-pct peak pelvic force |F| [N] | 150 | 150 |
| 99th-pct peak force rate |dF/dt| [N/s] | 8719 | 9622 |
| 99th-pct peak pelvis velocity during L1 [m/s] | - | 0.81 |
| max over trials of peak |F| [N] | 150 | |

Note: the commanded force is saturated at F_max = 150 N by the arbiter (above that the harness catch is requested),
so the 99th percentile sits at or just below the limit by construction; it is a design check (the SEA must deliver
F_max), not a free estimate of demand.

## Ball-screw SEA (per side, AP or ML axis), lead 5 mm, eta = 0.9
- Motor torque tau = F * lead / (2 pi eta): at F_max = 150 N -> **0.133 N m**; at the 99th-pct force
  (150 N, L1-active) -> 0.133 N m.
- Screw/motor speed for the 99th-pct pelvis velocity during L1 (0.81 m/s) if the carriage must follow the pelvis
  rigidly: 9683 rpm (n = v/lead = 161 rev/s). That is high for a 5 mm lead; the series spring
  decouples the motor from the pelvis (the motor only has to move the spring end), so this is an upper bound.
- Peak mechanical power at F_max and v = 0.81 m/s: 121 W per axis (upper bound, same assumption).
- Spring stiffness for 10 mm deflection at F_max: k = 150 N / 0.010 m = **15000 N/m**.
- Force-rate demand: 99th-pct |dF/dt| = 9622 N/s (L1-active) -> spring-end speed 641 mm/s
  (7697 rpm on the screw), which the motor must supply in addition to nothing else (SEA deflection rate).
  The sim does not resolve force-control bandwidth; the SPEC value (>= 5 Hz) is taken as given.

## Catch winch
- Static holding requirement 1.5 m g at m = 85 kg: **1251 N** (128 kgf).
- Simulated harness force (spring-damper, K_c = 4000 N/m, D_c = 800 N s/m): 99th-pct peak over the
  814 CAUGHT trials = 450 N; max = 491 N
  (below the static 1.5 m g requirement, i.e. the dynamic
  catch peak does not drive the winch/brake rating).
