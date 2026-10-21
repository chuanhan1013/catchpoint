"""Impaired-G1 push-recovery environment for PPO fine-tuning (no walker, no controller).

Same physics / PD / impairment conventions as g1cp.Sim: right-leg torque limits x0.5, observation delay 2 policy ticks,
action delay 1 tick (60 ms total).  Random pelvic pushes (0.1 s pulses), mostly lateral toward the weak side."""
import os
import sys
from collections import deque

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from g1cp import DEC, DT, KDS, KPS, PERIOD, Q0, TAU_NOM, gravity_orientation, scene_xml  # noqa: E402

N_OBS, N_CRIT = 47, 53
EP_LEN = 500                 # policy steps (10 s)


class PushEnv:
    def __init__(self, seed=0, weak=0.5, nd_obs=2, nd_act=1, push_range=(60.0, 300.0), p_weak_side=0.75):
        self.m = mujoco.MjModel.from_xml_path(scene_xml())
        self.m.opt.timestep = DT
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.tau_lim = TAU_NOM.copy()
        self.tau_lim[6:12] *= weak
        self.nd_obs, self.nd_act = nd_obs, nd_act
        self.push_range, self.p_weak = push_range, p_weak_side
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[7:] = Q0
        mujoco.mj_forward(self.m, self.d)
        self.k = 0                      # policy step counter
        self.obs_q = deque()
        self.act_q = deque()
        self.act = np.zeros(12, np.float32)
        self.tgt = Q0.copy()
        self.next_push = self.rng.uniform(1.0, 2.5)
        self.push_until, self.push_F = -1.0, np.zeros(2)
        self.last_obs = self._raw_obs()
        return self._actor_obs(self.last_obs), self._critic_obs()

    def t(self):
        return self.k * DT * DEC

    def _raw_obs(self):
        d = self.d
        ph = (self.t() % PERIOD) / PERIOD
        o = np.zeros(N_OBS, np.float32)
        o[:3] = d.qvel[3:6] * 0.25
        o[3:6] = gravity_orientation(d.qpos[3:7])
        o[9:21] = d.qpos[7:] - Q0
        o[21:33] = d.qvel[6:] * 0.05
        o[45:47] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
        return o

    def _actor_obs(self, raw):
        self.obs_q.append(raw)
        o = (self.obs_q[0] if len(self.obs_q) <= self.nd_obs else self.obs_q.popleft()).copy()
        o[33:45] = self.act
        return o

    def _critic_obs(self):
        d = self.d
        R = d.xmat[self.pid].reshape(3, 3)
        vb = R.T @ d.qvel[:3]
        c = np.zeros(N_CRIT, np.float32)
        c[:47] = self.last_obs
        c[33:45] = self.act
        c[47:50] = vb
        c[50] = d.qpos[2] - 0.78
        c[51:53] = self.push_F / 300.0 if self.t() < self.push_until else 0.0
        return c

    # ------------------------------------------------------------------
    def step(self, action):
        """action: raw policy output (12,).  Returns actor_obs, critic_obs, reward, done, info."""
        d = self.d
        prev = self.act
        self.act = np.asarray(action, np.float32)
        self.act_q.append(self.act.copy())
        if len(self.act_q) > self.nd_act:
            self.tgt = self.act_q.popleft() * 0.25 + Q0
        t = self.t()
        if t >= self.next_push:
            if self.rng.random() < self.p_weak:
                ang = -np.pi / 2 + self.rng.uniform(-0.35, 0.35)
            else:
                ang = self.rng.uniform(-np.pi, np.pi)
            mag = self.rng.uniform(*self.push_range)
            self.push_F = mag * np.array([np.cos(ang), np.sin(ang)])
            self.push_until = t + 0.1
            self.next_push = t + self.rng.uniform(2.0, 3.5)
        tau2 = 0.0
        for i in range(DEC):
            tt = t + i * DT
            tau = np.clip((self.tgt - d.qpos[7:]) * KPS - d.qvel[6:] * KDS, -self.tau_lim, self.tau_lim)
            d.ctrl[:] = tau
            tau2 += float(tau @ tau)
            d.xfrc_applied[self.pid, :2] = self.push_F if tt < self.push_until else 0.0
            mujoco.mj_step(self.m, d)
        self.k += 1
        self.last_obs = self._raw_obs()
        g = self.last_obs[3:6]
        z = d.qpos[2]
        fell = z < 0.55 or g[2] > -np.cos(np.radians(45))
        # reward
        v = d.qvel[:2]
        r = 1.0
        r -= 2.0 * float(g[0] ** 2 + g[1] ** 2)
        r -= 5.0 * max(0.0, 0.74 - z) ** 2 * 10
        r -= 0.1 * float(v @ v)
        r -= 0.01 * float(np.sum((self.act - prev) ** 2))
        r -= 2e-6 * tau2 / DEC
        timeout = self.k >= EP_LEN
        if fell:
            r -= 50.0
        done = fell or timeout
        aobs = self._actor_obs(self.last_obs)
        cobs = self._critic_obs()
        return aobs, cobs, r, done, dict(fell=fell, timeout=timeout and not fell)
