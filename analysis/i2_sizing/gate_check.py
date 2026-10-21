"""Load check of the hinged FRONT GATE ARM that carries the front-centre (CF) cable anchor (contract: front_gate).

The gate is a horizontal rectangular aluminium tube at pelvis height, hinged on the front-right column and latched to
the front-left column.  When latched it is a simply supported beam (span = hinge-latch distance) loaded at mid-span by the
horizontal CF cable force; the cable line of action intersects the arm axis, so there is no torsion.
Design cable force = max(T_slip centre, worst QP tension on frontC over results.json arrest_rows and nudge_rows).
Geometry, section, material and assumptions come from data/d3_contract.json (front_gate); results are written to
data/gate.json (raw floats; published numbers are rounded by make_report.py through publish.py).

Usage:  python analysis/i2_sizing/gate_check.py          (rewrite data/gate.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sizing as S  # noqa: E402

DATA = os.path.join(HERE, "data")
GATE_JSON = os.path.join(DATA, "gate.json")
RHO_AL = 2700.0          # kg/m3, aluminium


def design_loads(K, R):
    """Cable forces (N) the gate must carry, and the latch rating (1.5 x the design force)."""
    t_slip = float(K["T_slip"]["centre"])
    t_arrest = max(float(a["T"]["frontC"]) for a in R["arrest_rows"])
    t_nudge = max(float(a["T"]["frontC"]) for a in R["nudge_rows"])
    t_qp = max(t_arrest, t_nudge)
    design = max(t_slip, t_qp)
    return dict(T_slip_centre=t_slip, T_qp_frontC_arrest=t_arrest, T_qp_frontC_nudge=t_nudge, T_qp_frontC_worst=t_qp,
                design_cable_force=design, latch_rating=1.5 * design)


def compute():
    K = S.load_contract()
    with open(os.path.join(DATA, "results.json"), encoding="utf8") as f:
        R = json.load(f)
    g = K["front_gate"]
    tb = g["tube"]
    hinge, latch = np.array(g["hinge"], float), np.array(g["latch"], float)
    L = float(np.linalg.norm(latch - hinge))
    assert abs(L - g["arm_length"]) < 1e-9
    assert np.allclose(hinge[[0, 2]], latch[[0, 2]])                     # arm axis is along y
    a_mid = g["spool_position_along_arm"] * L
    # cable direction from the analysis layout (belt attachment -> anchor); force on the gate is -T u
    lay = S.d3_layout()
    _, U, _ = S.cable_dirs(lay)
    i = K["anchor_names"]["centre"][1]
    names = [n for grp in ("corner", "lateral", "centre") for n in K["anchor_names"][grp]]
    u = U[names.index(i)]
    assert np.allclose(lay.anchors[names.index(i)], K["centre_anchors"][1])
    assert abs(u[2]) < 1e-9 and abs(u[1]) < 1e-9 and u[0] > 0.999999     # horizontal, along +x (belt -> anchor)
    loads = design_loads(K, R)
    P = loads["design_cable_force"]
    # section (mm): outer b (horizontal x) x h (vertical z), wall t; strong axis vs the horizontal pull
    b, h, t = float(tb["width_mm"]), float(tb["height_mm"]), float(tb["wall_mm"])
    bi, hi = b - 2 * t, h - 2 * t
    A = b * h - bi * hi
    Iz = (h * b ** 3 - hi * bi ** 3) / 12.0                              # bending about the vertical axis (horizontal load)
    Sz = Iz / (b / 2)
    Iy = (b * h ** 3 - bi * hi ** 3) / 12.0                              # bending about the horizontal axis (weight)
    Sy = Iy / (h / 2)
    d = float(g["fairlead_hole_mm"])
    dI = 2 * d * t * (b / 2 - t / 2) ** 2                                # fairlead hole through both x-facing walls
    Iz_net = Iz - dI
    Sz_net = Iz_net / (b / 2)
    E = float(tb["E_GPa"]) * 1e3                                         # MPa
    sy = float(tb["sigma_y_MPa"])
    s_allow = sy / float(tb["allow_divisor"])
    # loads (N, m)
    w_arm = A * 1e-6 * RHO_AL * S.G                                      # N per m of tube
    W_sp = float(g["spool_mass_kg_assumed"]) * S.G
    # latched: simply supported span L, point load P at a_mid (= mid-span), weight W_sp at a_mid, w_arm distributed
    assert abs(a_mid - L / 2) < 1e-9
    R_latch_h = P / 2
    R_hinge_h = P / 2
    R_v = (W_sp + w_arm * L) / 2
    M_h_mid = P * L / 4                                                  # N m (horizontal bending, about z)
    M_v_mid = W_sp * L / 4 + w_arm * L ** 2 / 8                          # N m (weight, about the horizontal axis)
    sig_net = M_h_mid * 1e3 / Sz_net + M_v_mid * 1e3 / Sy
    sig_peak = float(g["hole_Kt"]) * M_h_mid * 1e3 / Sz_net + M_v_mid * 1e3 / Sy      # Kt on the hole flange stress
    sig_gross = M_h_mid * 1e3 / Sz + M_v_mid * 1e3 / Sy
    defl = P * (L * 1e3) ** 3 / (48 * E * Iz)                            # mm, at the cable exit (gross section)
    # unlatched (information only): cantilever from the hinge, load at mid-span
    M_hinge_cant = P * (L / 2)
    sig_cant = M_hinge_cant * 1e3 / Sz                                  # no hole at the hinge end
    out = dict(
        source="analysis/i2_sizing/gate_check.py from data/d3_contract.json (front_gate) and data/results.json",
        loads=loads,
        cable_direction_belt_to_anchor=[float(v) for v in u],
        span_m=L,
        section=dict(A_mm2=A, I_strong_mm4=Iz, S_strong_mm3=Sz, I_weak_mm4=Iy, S_weak_mm3=Sy, hole_dI_mm4=dI,
                     I_strong_net_mm4=Iz_net, S_strong_net_mm3=Sz_net, mass_per_m_kg=A * 1e-6 * RHO_AL,
                     arm_mass_kg=A * 1e-6 * RHO_AL * L),
        material=dict(sigma_y_MPa=sy, sigma_allow_MPa=s_allow, E_MPa=E),
        latched=dict(hinge_reaction_horizontal_N=R_hinge_h, latch_reaction_horizontal_N=R_latch_h,
                     vertical_reaction_each_N=R_v, hinge_reaction_resultant_N=float(np.hypot(R_hinge_h, R_v)),
                     latch_reaction_resultant_N=float(np.hypot(R_latch_h, R_v)),
                     M_hinge_Nm=0.0, M_mid_horizontal_Nm=M_h_mid, M_mid_vertical_Nm=M_v_mid,
                     sigma_gross_MPa=sig_gross, sigma_net_MPa=sig_net, sigma_peak_MPa=sig_peak,
                     SF_yield=sy / sig_peak, SF_allow=s_allow / sig_peak, deflection_mm=defl),
        unlatched_cantilever_info=dict(hinge_reaction_horizontal_N=P, M_hinge_Nm=M_hinge_cant, M_mid_Nm=0.0,
                                       sigma_hinge_MPa=sig_cant, SF_yield=sy / sig_cant,
                                       note="out of scope: the interlock disarms the system if the latch is open, and the "
                                            "latch is rated >= 1.5 x the design cable force"),
        pass_=bool(sig_peak <= s_allow and defl < 1.0),
    )
    out["pass"] = out.pop("pass_")
    return out


def main():
    out = compute()
    with open(GATE_JSON, "w", encoding="utf8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    la = out["latched"]
    print("design force %.1f N | hinge %.1f N latch %.1f N | M_mid %.2f N m | sigma_peak %.1f MPa | SF_yield %.2f | defl %.3f mm | pass=%s"
          % (out["loads"]["design_cable_force"], la["hinge_reaction_resultant_N"], la["latch_reaction_resultant_N"],
             la["M_mid_horizontal_Nm"], la["sigma_peak_MPa"], la["SF_yield"], la["deflection_mm"], out["pass"]))


if __name__ == "__main__":
    main()
