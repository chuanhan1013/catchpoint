import numpy as np
import pytest

from catchpoint.controllers import B0, B1
from catchpoint.montecarlo import make_population
from catchpoint.model import simulate
from i3 import emken, learner as L
from i3.skill_sim import default_skill, simulate_skill


def test_default_skill_reproduces_original_engine():
    pop = make_population(n_pat=4, n_push=10)
    pop["skill"] = default_skill(len(pop["m"]))
    for ctrl in (B0(), B1(0.05)):
        a, b = simulate(pop, ctrl), simulate_skill(pop, ctrl)
        assert (a["outcome"] == b["outcome"]).all()
        assert np.allclose(a["impulse"], b["impulse"])


def test_reward_effort_term_and_fall_penalty():
    res = dict(outcome=np.array([0, 2, 3]), step_len=np.array([0.4, 0.0, 0.3]), ankle_int=np.array([0.1, 0.1, 0.0]))
    r0 = L.rewards(res, 0.0)
    assert list(r0) == [1.0, 1.0, -1.0]          # CAUGHT counts as not fallen
    r1 = L.rewards(res, 0.5)
    assert np.allclose(r0 - r1, 0.5 * L.effort_of(res))


@pytest.fixture(scope="module")
def trained():
    groups = L.make_groups(1)[:1]                 # one mild patient
    H = L.train(groups, [0.0, 0.5, 3.0], lambda: B0(), n_sess=8, probe_every=8, probe_n=40)
    return groups, H


def test_learner_improves_without_robot(trained):
    groups, H = trained
    p = H["probe"]                                # rows: session 0, session 8
    assert p[-1][0] > p[0][0] + 0.05, p          # error-only learner (lambda=0) improves with budget


def test_lambda_changes_effort(trained):
    groups, H = trained
    z = H["z"][-1]
    assert not np.allclose(z[0], z[2])            # reward weighting changes the learned policy
    eff = H["effort"].mean(0)
    assert eff[2] < eff[0], eff                   # strong effort penalty -> less effortful behaviour


def test_step_per_session_is_limited(trained):
    _, H = trained
    steps = np.linalg.norm(np.diff(H["z"], axis=0), axis=2)
    assert steps.max() <= L.STEP_MAX + 1e-9


def test_emken_dependence_prediction():
    seeds = list(range(8)); x0 = np.full(8, 0.15)
    ret = {c: emken.run(c, x0, seeds)["retention_self"].mean() for c in ("B0", "B2", "CP")}
    assert ret["B0"] > ret["B2"]                  # always-on assistance -> worse unassisted retention
