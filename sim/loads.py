#!/usr/bin/env python3
"""Per-joint load in four situations, against XL430 and smaller servos.

    python3 sim/loads.py --variant v9_yaw_roll_display [--compare v8_yaw_roll]

Situations: standing still, this variant's best gait (walk.json), a fixed
reference gait (same for every variant so they can be compared), the demo.
When the best gait uses the IMU feedback balance, standing and the demo use it
too (the robot always has its IMU).  The walking verdict counts standing and the
two gaits; the demo (scripted bows and twists) is reported on its own.
Per joint: peak and RMS torque (N.m) and peak speed.  Pass criteria are
assumptions, stated in the output: peak below PEAK_LIMIT of stall (headroom for
landing impacts and control lag), RMS below RMS_LIMIT of stall (heat).
"""
import json
import os

import mujoco
import numpy as np

import balance
import demo
import variants
import walk

MOTORS = {                      # stall torque N.m, no-load rad/s, grams (ROBOTIS e-Manual)
    "XL430-W250-T (11.1V)": (1.40, 5.97, 57.2),
    "XC330-T288-T (11.1V)": (0.92, 6.81, 23.0),
    "XC330-M288-T (5V)":    (0.93, 8.48, 23.0),
    "XC330-T181-T (11.1V)": (0.76, 10.89, 23.0),
    "XL330-M288-T (5V)":    (0.52, 10.79, 18.0),
    "STS3032 (6V)":         (0.44, 11.6, 25.0),     # Feetech, 4.5 kg.cm, 0.09 s/60 deg
    "SCS0009 (6V)":         (0.226, 10.5, 13.2),    # Feetech, 2.3 kg.cm, 0.1 s/60 deg
}
PEAK_LIMIT, RMS_LIMIT = 0.70, 0.35
REF_GAIT = dict(freq=0.9, roll=6, swing=16, phase=90, pair="same")   # yaw-hip variants
REF_GAIT_PITCH = dict(freq=0.4, roll=6, swing=4.5, phase=330, pair="anti")               # v7
REF_GAIT_PITCH3 = dict(freq=0.7, roll=6, swing=3, phase=300, pair="anti", follow=1.0)  # hip + ankle pitch


def record(variant, drive, t_end, t0=2.0):
    m = mujoco.MjModel.from_xml_path(variants.scene(variant))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    tq, vel, fell = [], [], None
    z0 = None
    while d.time < t_end:
        drive(m, d)
        mujoco.mj_step(m, d)
        z0 = z0 or d.subtree_com[1][2]
        if d.subtree_com[1][2] < 0.75 * z0:
            fell = round(d.time, 1)
            break
        if d.time >= t0:
            tq.append(np.abs(d.actuator_force.copy()))
            vel.append(np.abs(d.actuator_velocity.copy()))
    tq, vel = np.array(tq), np.array(vel)
    out = {n: dict(peak=float(tq[:, i].max()), rms=float(np.sqrt((tq[:, i] ** 2).mean())),
                   p95=float(np.percentile(tq[:, i], 95)), speed=float(vel[:, i].max()))
           for i, n in enumerate(names)}
    return out, fell, float(m.body_mass.sum())


def gait_driver(g, swing):
    state = {}

    def drive(m, d):
        tg = walk.targets(d.time, g, swing)
        if g.get("fb"):                          # the same feedback balance as the gait search
            if "fb" not in state:
                state["fb"] = balance.make(m, g)
            state["fb"].update(d, tg, m.opt.timestep)
        for k, v in tg.items():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
            if i >= 0:
                d.ctrl[i] = v
    return drive


def posed_driver(pose, fb):
    """A scripted pose (pose(t) -> targets, or None for standing still) with the
    feedback balance on top."""
    state = {}

    def drive(m, d):
        tg = dict(pose(d.time)) if pose else {}
        if fb:
            if "fb" not in state:
                state["fb"] = balance.make(m, {"fb": fb})
            state["fb"].update(d, tg, m.opt.timestep)
        for k, v in tg.items():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
            if i >= 0:
                d.ctrl[i] = v
    return drive


def analyse(variant):
    m = mujoco.MjModel.from_xml_path(variants.scene(variant))
    swing = walk.swing_joints(m)
    has_ankle_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_pitch_l") >= 0
    ref = REF_GAIT if swing == "hip_yaw" else (REF_GAIT_PITCH3 if has_ankle_pitch else REF_GAIT_PITCH)
    best = walk.best_gait(variant)
    if best and best.get("fb"):
        ref = dict(ref, fb=best["fb"])
    demo.configure(variant)
    fb = best.get("fb") if best else None
    runs = {
        "stand": record(variant, posed_driver(None, fb), 5.0),
        **({"best gait": record(variant, gait_driver(best, swing), walk.T_END)} if best else {}),
        "reference gait": record(variant, gait_driver(ref, swing), walk.T_END),
        "demo": record(variant, posed_driver(demo.targets_at, fb), demo.DURATION),
    }
    return dict(best=best, ref=ref, runs=runs)


def report(variant, res):
    runs = res["runs"]
    mass = next(iter(runs.values()))[2]
    L = [f"== {variant}  ({mass:.2f} kg)",
         f"best gait {res['best']}", f"reference gait {res['ref']}", ""]
    joints = list(runs["stand"][0])
    L.append(f"{'joint':13s} " + " ".join(f"{k:>21s}" for k in runs) + "   max speed")
    L.append(f"{'':13s} " + " ".join(f"{'peak / rms  N.m':>21s}" for _ in runs) + "   rad/s")
    for j in joints:
        cells = [f"{r[0][j]['peak']:9.3f} / {r[0][j]['rms']:.3f}" for r in runs.values()]
        L.append(f"{j:13s} " + " ".join(f"{c:>21s}" for c in cells)
                 + f"   {max(r[0][j]['speed'] for r in runs.values()):5.2f}")
    falls = {k: r[1] for k, r in runs.items() if r[1] is not None}
    if falls:
        L.append("fell: " + ", ".join(f"{k} at {v} s" for k, v in falls.items()))
    tables, verdicts = {}, {}
    for title, keys in (("walking (standing, best gait, reference gait; IMU feedback on)",
                         [k for k in runs if k != "demo"]),
                        ("all runs, the demo's scripted bows and twists included", list(runs))):
        worst = {j: (max(runs[k][0][j]["peak"] for k in keys),
                     max(runs[k][0][j]["rms"] for k in keys),
                     max(runs[k][0][j]["speed"] for k in keys)) for j in joints}
        L += ["", f"motor check, {title}: peak under {PEAK_LIMIT:.0%} of stall, RMS under "
              f"{RMS_LIMIT:.0%} of stall, speed under no-load speed"]
        L.append(f"{'joint':13s} " + " ".join(f"{k.split(' ')[0]:>13s}" for k in MOTORS))
        verdict = {}
        for j in joints:
            pk, rms, sp = worst[j]
            cells = []
            for name, (stall, w0, _) in MOTORS.items():
                ok = pk <= PEAK_LIMIT * stall and rms <= RMS_LIMIT * stall and sp <= w0
                verdict.setdefault(name, {})[j] = ok
                cells.append(f"{pk/stall*100:4.0f}%/{rms/stall*100:3.0f}% {'ok' if ok else '--'}")
            L.append(f"{j:13s} " + " ".join(f"{c:>13s}" for c in cells))
        tables[title], verdicts[title] = worst, verdict
    L.append("  (cells: peak % of stall / RMS % of stall, then ok or --)")
    return "\n".join(L), tables, verdicts


def main():
    ap = variants.parser(__doc__)
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    for v in [a.variant] + a.compare:
        res = analyse(v)
        text, worst, verdict = report(v, res)
        print(text + "\n")
        with open(os.path.join(variants.folder(v), "loads.txt"), "w") as fh:
            fh.write(text + "\n")
        with open(os.path.join(variants.folder(v), "loads.json"), "w") as fh:
            json.dump(dict(worst=worst, verdict=verdict, best=res["best"], ref=res["ref"]), fh, indent=1)


if __name__ == "__main__":
    main()
