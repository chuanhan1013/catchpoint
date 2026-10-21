import numpy as np
import pytest

from catchpoint.controllers import B0, B1, CP
from catchpoint.model import (CAUGHT, FALL, SELF, G, barrier_h, build_pop, dist_to, lip_step,
                              make_hull, min_norm_force, project, simulate)
from catchpoint.montecarlo import make_population


def test_lip_free_fall_diverges_at_omega():
    om = np.sqrt(G / 0.9)
    x = np.array([[0.01, -0.02]]); v = np.array([[0.03, 0.01]])
    p = np.zeros((1, 2)); F = np.zeros((1, 2)); m = np.array([70.0]); omg = np.array([om])
    xi0 = x + v / om
    dt = 0.002
    for k in range(1, 501):
        x, v = lip_step(x, v, p, F, m, omg, dt)
        if k in (100, 250, 500):
            ratio = np.linalg.norm(x + v / om - p) / np.linalg.norm(xi0 - p)
            assert ratio == pytest.approx(np.exp(om * k * dt), rel=1e-9)


def test_xi_constant_when_p_equals_xi_and_no_force():
    om = np.sqrt(G / 0.95)
    x = np.array([[0.02, 0.05]]); v = np.array([[0.1, -0.2]])
    xi0 = x + v / om
    for _ in range(1000):
        xi = x + v / om
        x, v = lip_step(x, v, xi, np.zeros((1, 2)), np.array([70.0]), np.array([om]), 0.002)
    assert np.abs(x + v / om - xi0).max() < 1e-9


def test_min_norm_force_hits_barrier_exactly():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 2)) * 0.1
    land0 = a + rng.normal(size=(200, 2)) * 0.6
    r = rng.uniform(0.1, 0.5, 200)
    beta = rng.uniform(0.005, 0.05, 200)
    F = min_norm_force(land0, a, r, beta)
    h = barrier_h(F, land0, a, r, beta)
    h0 = barrier_h(np.zeros_like(F), land0, a, r, beta)
    viol = h0 < 0
    assert viol.sum() > 20
    assert np.abs(h[viol]).max() < 1e-6                     # h(F*) = 0
    assert np.abs(F[~viol]).max() == 0.0                    # already recoverable -> F = 0
    # minimality: any other feasible force is at least as large
    for _ in range(50):
        G_ = rng.normal(size=F.shape) * 80
        ok = barrier_h(G_, land0, a, r, beta) >= 0
        assert np.all(np.linalg.norm(G_[ok], axis=1) >= np.linalg.norm(F[ok], axis=1) - 1e-9)


def test_hull_projection_matches_bruteforce():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(3)
    for _ in range(30):
        c1 = rng.uniform(-0.2, 0.2, 2); c2 = rng.uniform(-0.2, 0.2, 2)
        h1 = rng.uniform(0.02, 0.11, 2); h2 = rng.uniform(0.02, 0.11, 2)
        S = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        pts = np.vstack([c1 + S * h1, c2 + S * h2])
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        P = rng.uniform(-0.5, 0.5, (40, 2))
        got = project(P, make_hull(np.tile(c1, (40, 1)), np.tile(h1, (40, 1)),
                                   np.tile(c2, (40, 1)), np.tile(h2, (40, 1))))
        for p, g in zip(P, got):
            best = np.inf
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
                best = min(best, np.linalg.norm(p - (a + t * (b - a))))
            inside = np.all(hull.equations[:, :2] @ p + hull.equations[:, 2] <= 1e-12)
            want = 0.0 if inside else best
            assert np.linalg.norm(p - g) == pytest.approx(want, abs=2e-3)


def _run(ctrl, sev, par, dv):
    pop = build_pop([sev], par, 72.0, 0.92, [dv])
    return simulate(pop, ctrl)["outcome"][0]


def test_sanity_a_b0_small_push_is_self():
    assert _run(B0(), 0.0, 0, [0.2, 0.0]) == SELF


def test_sanity_b_b0_large_push_paretic_side_falls():
    assert _run(B0(), 1.0, 0, [0.0, 0.9]) == FALL     # leg 0 (left) is paretic, +y = left


def test_sanity_c_catchpoint_prevents_fall_of_b():
    assert _run(CP(0.03), 1.0, 0, [0.0, 0.9]) != FALL


def test_sanity_d_b1_low_threshold_catches_small_push():
    assert _run(B1(0.03), 0.0, 0, [0.2, 0.0]) == CAUGHT


def test_population_is_reproducible_and_paired():
    a, b = make_population(), make_population()
    assert len(a["m"]) == 2000
    assert np.array_equal(a["dv"], b["dv"]) and np.array_equal(a["Lmax"], b["Lmax"])


def test_cp_sat_is_identical_to_arbiter():
    """h(F_max u) < 0  <=>  |F*| > F_max, so a saturate-then-check variant equals the implemented arbiter."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(500, 2)) * 0.1
    land0 = a + rng.normal(size=(500, 2)) * 0.6
    r = rng.uniform(0.1, 0.5, 500); beta = rng.uniform(0.005, 0.05, 500)
    F = min_norm_force(land0, a, r, beta)
    mag = np.linalg.norm(F, axis=1)
    u = np.where(mag[:, None] > 0, F / np.maximum(mag, 1e-12)[:, None], 0)
    h_sat = barrier_h(150.0 * u, land0, a, r, beta)
    big = mag > 150.0
    assert big.sum() > 0 and (h_sat[big] < 0).all()            # saturated force still unrecoverable -> catch
    ok = (mag > 0) & ~big
    assert np.abs(barrier_h(F[ok], land0[ok], a[ok], r[ok], beta[ok])).max() < 1e-9   # F* itself reaches h = 0


def test_crossover_used_when_side_step_infeasible():
    from catchpoint.model import _choose_step
    # xi far on the left of both feet, left (paretic) leg reach tiny -> only the right leg can go: it must cross over
    om = np.array([3.3]); L = np.array([[0.2, 0.6]]); Ts = np.array([[0.6, 0.4]]); sc = np.ones((1, 2))
    fc = np.array([[[0.0, 0.1], [0.0, -0.1]]])
    leg, mode, tgt, T = _choose_step(np.array([[0.0, 0.30]]), fc, om, L, Ts, sc)
    assert leg[0] == 1 and mode[0] == 1 and tgt[0, 1] >= 0.1 + 0.09 - 1e-9 and T[0] == pytest.approx(0.5)


def test_cp_predictor_matches_plant_with_exact_lag():
    """Predicted XCoM (delay + first-order lag propagated exactly) equals the true XCoM Da ticks ahead,
    for a step command followed by a ramp, with constant CoP."""
    from catchpoint.controllers import CP
    from catchpoint.model import Config
    cfg = Config()
    pop = build_pop([0.4], 0, 70.0, 0.9, [[0.0, 0.0]])          # identified model == true model
    cp = CP(0.03); cp.init(pop, cfg, 1)
    Ds, Da, dt = cp.Ds, cp.Da, cfg.dt
    n = 500; off = Ds + Da + 10
    cmd = np.zeros((n + off, 1, 2))
    cmd[off + 100:, 0] = [50.0, -20.0]                           # step
    for t in range(200, n):
        cmd[off + t, 0] = [50.0, -20.0] + 0.5 * (t - 200) * np.array([1.0, 0.7])   # ramp
    alpha = 1 - np.exp(-dt / cfg.tau_lag)
    Fact = np.zeros((n + off, 1, 2)); F = np.zeros((1, 2))
    x = np.array([[0.02, -0.01]]); v = np.array([[0.1, 0.2]]); p = np.array([[0.03, 0.05]])
    om = pop["omega"]; mass = pop["m"]
    xs, vs = [], []
    for t in range(n):
        xs.append(x.copy()); vs.append(v.copy())
        F = F + alpha * (cmd[off + t - Da] - F)
        Fact[off + t] = F
        x, v = lip_step(x, v, p, F, mass, om, dt)
    worst = 0.0
    for k in range(60, n - Da):
        xi_m = xs[k - Ds] + vs[k - Ds] / om[:, None]
        pred = cp.predict_xi(xi_m, p, Fact, cmd, off, k)
        true = xs[k + Da] + vs[k + Da] / om[:, None]
        worst = max(worst, np.abs(pred - true).max())
    assert worst < 1e-9


def test_crossover_invalid_candidate_never_selected():
    from catchpoint.model import candidate_barrier
    # xi far to the left of the stance feet: right leg side-step would have to land left of the left foot
    # (cross-over, invalid as side-step); make sure the chosen candidate is crossover-valid.
    fc = np.array([[[0.0, 0.1], [0.0, -0.1]]])
    cb = candidate_barrier(np.array([[0.0, 0.5]]), fc, -np.ones(1, int), np.zeros(1, int), np.zeros(1), np.zeros(1),
                           np.array([3.3]), np.array([70.0]), np.array([[0.5, 0.5]]), np.array([[0.4, 0.4]]),
                           np.ones((1, 2)), 0.03)
    assert cb["anyvalid"][0]
    jb = cb["jb"][0]; mode, leg = jb // 2, jb % 2
    sd = (1.0, -1.0)[leg] * (cb["land0"][0, 1] - cb["a"][0, 1])
    assert (sd >= 0.08) if mode == 0 else (sd <= -0.09)
