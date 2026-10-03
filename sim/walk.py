#!/usr/bin/env python3
"""Open-loop rocking gait (Otto / Zowi style) and a search for the best one.

Joints are driven by sine oscillators at one frequency f:
    ankle rolls : A_roll * sin(wt)                  rock onto one foot, then the other
    hip swing   : A_swing * sin(wt + phase) on the left hip, and on the right hip
                  either the opposite ("anti", legs scissor) or the same ("same")
The swing joints are the hip pitch joints (v7) or hip yaw joints when a
variant has them.  Otto's walk turns both hip yaw servos the same way: the
stance foot stays put, the pelvis turns about it, and the lifted foot slides
forward on an arc without turning toward the other foot.

    python3 sim/walk.py --variant v7_roll_only --search     # grid search, saves walk.json
    python3 sim/walk.py --variant v7_roll_only --video walk.mp4   # replay the best gait
"""
import itertools
import json
import os
import time
from multiprocessing import Pool

import mujoco
import numpy as np

import variants

RAMP, T_END = 2.0, 14.0          # amplitude ramp, total time (s); distance counted after the ramp


def swing_joints(m):
    for pre in ("hip_yaw", "hip_pitch"):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{pre}_l") >= 0:
            return pre
    raise RuntimeError("no hip joints to swing")


def targets(t, g, swing):
    a = min(1.0, t / RAMP)
    w = 2 * np.pi * g["freq"]
    roll = a * np.deg2rad(g["roll"]) * np.sin(w * t)
    sw = a * np.deg2rad(g["swing"]) * np.sin(w * t + np.deg2rad(g["phase"]))
    sign = 1.0 if g.get("pair", "anti") == "same" else -1.0
    out = {"ankle_roll_l": roll, "ankle_roll_r": roll, f"{swing}_l": sw, f"{swing}_r": sign * sw}
    if g.get("follow") is not None:          # ankle pitch counters the hip: sole stays level
        out["ankle_pitch_l"] = -g["follow"] * sw
        out["ankle_pitch_r"] = -g["follow"] * sign * sw
    return out


def roll_of(xmat):
    R = xmat.reshape(3, 3)
    return np.degrees(np.arctan2(R[2, 1], R[2, 2]))


def _p2p_detrended(t, x):
    """Peak-to-peak of x after removing its straight-line trend."""
    if len(t) < 3:
        return 0.0
    k = np.polyfit(t, x, 1)
    r = np.asarray(x) - np.polyval(k, t)
    return float(r.max() - r.min())


def yaw_of(xmat):
    R = xmat.reshape(3, 3)
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def run(scene, g, record=None):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    swing = swing_joints(m)
    ids = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
           for k in targets(0, g, swing)}
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    z0 = d.subtree_com[1][2]
    tor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    start, fell, peak = None, None, 0.0
    trace = []
    while d.time < T_END:
        for k, v in targets(d.time, g, swing).items():
            d.ctrl[ids[k]] = v
        mujoco.mj_step(m, d)
        if start is None and d.time >= RAMP:
            start = (d.xpos[pel].copy(), yaw_of(d.xmat[pel]))
        peak = max(peak, np.abs(d.actuator_force).max())
        if start is not None:
            trace.append((d.time, d.xpos[pel][1], roll_of(d.xmat[tor]), yaw_of(d.xmat[pel])))
        if d.subtree_com[1][2] < 0.75 * z0:
            fell = round(d.time, 2)
            break
        if record is not None:
            record(m, d)
    end = d.xpos[pel]
    dist = end - start[0] if start is not None else np.zeros(3)
    dur = max(d.time - RAMP, 1e-6)
    tr = np.array(trace) if trace else np.zeros((0, 4))
    wob = {}
    if len(tr) > 10:
        wob = dict(sway_mm=_p2p_detrended(tr[:, 0], tr[:, 1] * 1000),     # side-to-side of the pelvis
                   roll_deg=float(tr[:, 2].max() - tr[:, 2].min()),        # torso rocking
                   yaw_osc_deg=_p2p_detrended(tr[:, 0], tr[:, 3]))         # heading wiggle
    return dict(g, fell=fell, dx=dist[0] * 1000, dy=dist[1] * 1000,
                dyaw=(yaw_of(d.xmat[pel]) - start[1]) if start else 0.0,
                speed=dist[0] * 1000 / dur, peak=peak / 1.4 * 100, **wob)


STEP_RATIO_MIN = 0.6    # at least this share of foot travel must happen in the air
PEAK_MAX = 70.0         # % of XL430 stall: faster gaits that ride the torque limit are
                        # not usable on hardware (added after v8; v7's walk.json predates it)


def foot_stats(scene, g):
    """How much the feet really step: time in the air, lift height, and how much
    of each foot's forward travel happens in the air rather than sliding."""
    import analyse as A
    log = []

    def rec(m, d):
        if d.time < RAMP:
            return
        W = m.body_mass.sum() * 9.81
        row = [d.time]
        for f in ("foot_l", "foot_r"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f)
            row += [A.foot_load(m, d, f) / W, A.foot_world_verts(m, d, f)[:, 2].min(), d.xpos[b][0]]
        log.append(row)

    r = run(scene, g, record=rec)
    L = np.array(log)
    out = dict(r)
    air_t, lift, slip, step = [], [], 0.0, 0.0
    for load, z, x in ((1, 2, 3), (4, 5, 6)):
        air = L[:, load] < 0.01
        air_t.append(air.mean())
        lift.append(L[:, z].max() * 1000)
        dx = np.abs(np.diff(L[:, x])) * 1000
        slip += dx[L[1:, load] > 0.05].sum()
        step += dx[air[1:]].sum()
    out.update(air_time=float(np.mean(air_t)), lift_mm=float(np.max(lift)),
               step_ratio=float(step / max(step + slip, 1e-9)))
    return out


def _job(args):
    scene, g = args
    return run(scene, g)


def _stats_job(args):
    scene, r = args
    return foot_stats(scene, {k: r[k] for k in ("freq", "roll", "swing", "phase", "pair", "follow") if k in r})


GRID = dict(freq=(0.5, 0.7, 0.9, 1.1), roll=(4, 6, 8), swing=(3, 6, 9, 12, 16, 20),
            phase=(0, 90, 180, 270), pair=("anti", "same"))
# second pass: slow rocking, small swings, fine phase
GRID_FINE = dict(freq=(0.3, 0.4, 0.5, 0.6), roll=(3, 4, 5, 6, 7), swing=(1.5, 3, 4.5, 6),
                 phase=tuple(range(0, 360, 30)), pair=("anti", "same"))
# third pass: faster and bigger hip swings, around Otto's own setting (90 deg, same)
GRID_WIDE = dict(freq=(1.1, 1.3, 1.5, 1.8), roll=(3, 4, 5), swing=(20, 25, 30),
                 phase=(60, 75, 90, 105, 120), pair=("same",))


# hip pitch + ankle pitch + ankle roll (v10): legs scissor, the ankle pitch
# counters a share `follow` of the hip swing to keep the soles level
GRID_PITCH3 = dict(freq=(0.4, 0.5, 0.6, 0.7, 0.9, 1.1), roll=(3, 4, 5, 6, 7),
                   swing=(3, 6, 9, 12, 16), phase=tuple(range(0, 360, 30)), pair=("anti",),
                   follow=(0.0, 0.5, 1.0))


def grids_for(scene):
    m = mujoco.MjModel.from_xml_path(scene)
    has_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_pitch_l") >= 0
    if has_pitch and swing_joints(m) == "hip_pitch":
        return [GRID_PITCH3]
    return [GRID, GRID_FINE, GRID_WIDE]


def search(variant):
    scene = variants.scene(variant)
    grids = grids_for(scene)
    gaits = [dict(zip(G, vals)) for G in grids for vals in itertools.product(*G.values())]
    t = time.time()
    with Pool() as pool:
        res = pool.map(_job, [(scene, g) for g in gaits])
    ok = [r for r in res if r["fell"] is None]
    straight = sorted((r for r in ok if abs(r["dyaw"]) < 20 and r["peak"] <= PEAK_MAX),
                      key=lambda r: -r["dx"])
    # the furthest gaits often shuffle; check the top ones for real steps
    with Pool() as pool:
        checked = pool.map(_stats_job, [(scene, r) for r in straight[:40]])
    good = sorted((r for r in checked if r["step_ratio"] >= STEP_RATIO_MIN), key=lambda r: -r["dx"])
    lines = [f"variant {variant}: {len(gaits)} gaits, {len(ok)} stayed up, "
             f"{len(straight)} kept their heading with peak torque <= {PEAK_MAX:.0f}% of stall "
             f"({time.time()-t:.0f} s)",
             f"top 40 by distance checked for real steps; {len(good)} move their feet mostly "
             f"through the air (>= {STEP_RATIO_MIN:.0%} of foot travel), the rest mostly slide", "",
             f"{'freq':>5s} {'roll':>5s} {'swing':>6s} {'phase':>6s} {'hips':>5s} {'foll':>4s} | "
             f"{'forward':>8s} {'side':>6s} {'turn':>6s} {'speed':>9s} {'peak':>5s} | {'in air':>6s} "
             f"{'lift':>6s} {'stepped':>7s} | {'sway':>6s} {'rock':>5s} {'wiggle':>6s}"]
    fmt = lambda r: (f"{r['freq']:5.1f} {r['roll']:5.0f} {r['swing']:6.1f} {r['phase']:6.0f} "
                     f"{r.get('pair', 'anti'):>5s} {r.get('follow', 0) or 0:4.1f} | "
                     f"{r['dx']:6.0f}mm {r['dy']:4.0f}mm {r['dyaw']:4.0f}deg "
                     f"{r['speed']:5.1f}mm/s {r['peak']:4.0f}% | {r['air_time']*100:5.0f}% "
                     f"{r['lift_mm']:4.1f}mm {r['step_ratio']*100:6.0f}% | "
                     f"{r.get('sway_mm', 0):4.0f}mm {r.get('roll_deg', 0):4.1f}d {r.get('yaw_osc_deg', 0):5.1f}d")
    lines += [fmt(r) for r in good[:10]]
    calm = None
    if good:
        near = [r for r in good if r["dx"] >= 0.6 * good[0]["dx"]]
        calm = min(near, key=lambda r: r.get("sway_mm", 1e9))
        lines += ["", "calmest gait that still covers >= 60% of the best distance (least side-to-side sway):",
                  fmt(calm)]
    lines += ["", "sway = side-to-side travel of the pelvis, rock = torso roll range, "
              "wiggle = heading swing (all peak-to-peak, trend removed)"]
    lines += ["", "furthest overall (for comparison, may be sliding):"]
    lines += [fmt(r) for r in sorted(checked, key=lambda r: -r["dx"])[:3]]
    falls = {}
    for r in res:
        if r["fell"] is not None:
            falls.setdefault(r["roll"], 0)
            falls[r["roll"]] += 1
    lines.append("falls by roll amplitude: " + ", ".join(f"{k} deg: {v}" for k, v in sorted(falls.items())))
    text = "\n".join(lines)
    print(text)
    out = variants.folder(variant)
    with open(os.path.join(out, "walk.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out, "walk.json"), "w") as fh:
        json.dump(dict(best=good[0] if good else None, calm=calm, top=good[:10], grid=grids,
                       tested=len(gaits), stayed_up=len(ok), step_ratio_min=STEP_RATIO_MIN,
                       peak_max=PEAK_MAX),
                  fh, indent=2, default=float)
    return good


def best_gait(variant):
    with open(os.path.join(variants.folder(variant), "walk.json")) as fh:
        return json.load(fh)["best"]


def main():
    ap = variants.parser(__doc__)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--video")
    a = ap.parse_args()
    if a.search:
        search(a.variant)
    if a.video:
        import walk_video
        walk_video.render(a.variant, best_gait(a.variant), a.video)


if __name__ == "__main__":
    main()
