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

import balance
import variants

# worker processes for the search; the v19 models need ~0.7 GB each, so cap it on small
# machines with WALK_PROCS (default: one per CPU)
N_PROCS = int(os.environ.get("WALK_PROCS", os.cpu_count() or 1))

RAMP, T_END = 2.0, 14.0          # amplitude ramp, total time (s); distance counted after the ramp


def swing_joints(m):
    """hip pitch when there is one with a hip yaw (v17: the legs scissor, the yaw may
    add a twist), else whichever hip joint the variant has"""
    has = lambda pre: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{pre}_l") >= 0
    if has("hip_pitch") and has("hip_yaw"):
        return "hip_pitch"
    for pre in ("hip_yaw", "hip_pitch"):
        if has(pre):
            return pre
    raise RuntimeError("no hip joints to swing")


class HipCollision:
    """Leg / belly collision during a run, for variants whose hip seam MuJoCo cannot
    model (robot/<variant>/hip_range.json from tools/build_model.hip_range, checked
    on the real part volumes): counts the steps where either leg's (hip yaw, hip
    pitch) falls in a cell where the leg touches the belly, backpack or display.
    Leg vs leg and foot vs foot stay with MuJoCo's own contacts."""
    FIXED = ("belly", "backpack", "display")

    def __init__(self, m, scene):
        path = os.path.join(os.path.dirname(scene), "hip_range.json")
        self.ok = os.path.exists(path)
        if not self.ok:
            return
        h = json.load(open(path))
        self.yaw, self.pitch = np.array(h["yaw"], float), np.array(h["pitch"], float)
        self.bad = np.array([[any(n in self.FIXED for n in c) for c in row] for row in h["hits"]])
        def q(n):                                           # None: no such joint (v20 has no hip yaw)
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            return m.jnt_qposadr[j] if j >= 0 else None
        self.adr = {sd: (q(f"hip_yaw_{sd}"), q(f"hip_pitch_{sd}")) for sd in ("l", "r")}
        self.steps = self.hits = 0

    def update(self, d):
        if not self.ok:
            return
        self.steps += 1
        for sd, sgn in (("l", 1.0), ("r", -1.0)):          # the map is the left leg's: mirror the right yaw
            ya, pa = self.adr[sd]
            y = np.degrees(d.qpos[ya]) * sgn if ya is not None else 0.0
            p = np.degrees(d.qpos[pa])
            i = int(np.abs(self.yaw - y).argmin())
            j = int(np.abs(self.pitch - p).argmin())
            if abs(y) > self.yaw[-1] + 2.5 or self.bad[i, j]:
                self.hits += 1
                return

    def share(self):
        return self.hits / max(self.steps, 1) if self.ok else 0.0


def targets(t, g, swing):
    a = min(1.0, t / RAMP)
    w = 2 * np.pi * g["freq"]
    roll = a * np.deg2rad(g["roll"]) * np.sin(w * t)
    sw = a * np.deg2rad(g["swing"]) * np.sin(w * t + np.deg2rad(g["phase"]))
    sign = 1.0 if g.get("pair", "anti") == "same" else -1.0
    out = {"ankle_roll_l": roll, "ankle_roll_r": roll, f"{swing}_l": sw, f"{swing}_r": sign * sw}
    if g.get("yaw"):                         # v17: both hip yaws turn together with the swing (Otto)
        yw = a * np.deg2rad(g["yaw"]) * np.sin(w * t + np.deg2rad(g["phase"] + 90.0))
        out["hip_yaw_l"] = out["hip_yaw_r"] = yw
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


def run(scene, g, record=None, t_end=None):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    swing = swing_joints(m)
    ids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    fb = balance.make(m, g)                  # feedback balance, if the gait asks for it
    hc = HipCollision(m, scene)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    z0 = d.subtree_com[1][2]
    tor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    start, fell, peak = None, None, 0.0
    trace = []
    t_end = t_end or T_END
    while d.time < t_end:
        tg = targets(d.time, g, swing)
        if fb is not None:
            fb.update(d, tg, m.opt.timestep)
        for k, v in tg.items():
            if k in ids:
                d.ctrl[ids[k]] = v
        mujoco.mj_step(m, d)
        hc.update(d)
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
                speed=dist[0] * 1000 / dur, peak=peak / m.actuator_forcerange[:, 1].max() * 100,
                hip_hit=hc.share(), **wob)


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


def _long_job(args):
    scene, g, t = args
    return run(scene, g, t_end=t)


def _stats_job(args):
    scene, r = args
    return foot_stats(scene, {k: r[k] for k in ("freq", "roll", "swing", "phase", "pair", "follow", "yaw", "fb")
                              if k in r})


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


# hip yaw + hip pitch + ankle roll (v17): legs scissor on the hip pitch, the hip yaws
# may add Otto's twist a quarter period ahead
GRID_HIP2 = dict(freq=(0.5, 0.7, 0.9, 1.1, 1.3), roll=(3, 4, 5, 6, 7), swing=(0, 2, 4, 8, 12),
                 phase=tuple(range(0, 360, 30)), pair=("anti",), yaw=(0, 6, 12, 20))


def grids_for(scene):
    m = mujoco.MjModel.from_xml_path(scene)
    has_pitch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_pitch_l") >= 0
    has_yaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_yaw_l") >= 0
    if swing_joints(m) == "hip_pitch" and has_yaw and not has_pitch:
        return [GRID_HIP2]
    if has_pitch and swing_joints(m) == "hip_pitch":
        return [GRID_PITCH3]
    return [GRID, GRID_FINE, GRID_WIDE]


def search(variant, fb=None):
    scene = variants.scene(variant)
    grids = grids_for(scene)
    gaits = [dict(zip(G, vals)) for G in grids for vals in itertools.product(*G.values())]
    if fb:                                   # every gait runs with the feedback balance (balance.py)
        gaits = [dict(g, fb=fb) for g in gaits]
    t = time.time()
    with Pool(N_PROCS) as pool:
        res = pool.map(_job, [(scene, g) for g in gaits])
    ok = [r for r in res if r["fell"] is None]
    hit = sum(1 for r in ok if r.get("hip_hit", 0) > 0)
    ok = [r for r in ok if r.get("hip_hit", 0) == 0]     # the leg must never touch the belly
    straight = sorted((r for r in ok if abs(r["dyaw"]) < 20 and r["peak"] <= PEAK_MAX),
                      key=lambda r: -r["dx"])
    # the furthest gaits often shuffle; check the top ones for real steps
    with Pool(N_PROCS) as pool:
        checked = pool.map(_stats_job, [(scene, r) for r in straight[:40]])
    good = sorted((r for r in checked if r["step_ratio"] >= STEP_RATIO_MIN), key=lambda r: -r["dx"])
    lines = ([f"feedback balance on: {fb}"] if fb else []) + [f"variant {variant}: {len(gaits)} gaits, {len(ok)} stayed up, "
             f"{len(straight)} kept their heading with peak torque <= {PEAK_MAX:.0f}% of stall "
             f"({time.time()-t:.0f} s)"
             + (f"; {hit} more stayed up but hit the belly with a leg (dropped)" if hit else ""),
             f"top 40 by distance checked for real steps; {len(good)} move their feet mostly "
             f"through the air (>= {STEP_RATIO_MIN:.0%} of foot travel), the rest mostly slide", "",
             f"{'freq':>5s} {'roll':>5s} {'swing':>6s} {'phase':>6s} {'hips':>5s} {'foll':>4s} {'yaw':>4s} | "
             f"{'forward':>8s} {'side':>6s} {'turn':>6s} {'speed':>9s} {'peak':>5s} | {'in air':>6s} "
             f"{'lift':>6s} {'stepped':>7s} | {'sway':>6s} {'rock':>5s} {'wiggle':>6s}"]
    fmt = lambda r: (f"{r['freq']:5.1f} {r['roll']:5.0f} {r['swing']:6.1f} {r['phase']:6.0f} "
                     f"{r.get('pair', 'anti'):>5s} {r.get('follow', 0) or 0:4.1f} {r.get('yaw', 0) or 0:4.0f} | "
                     f"{r['dx']:6.0f}mm {r['dy']:4.0f}mm {r['dyaw']:4.0f}deg "
                     f"{r['speed']:5.1f}mm/s {r['peak']:4.0f}% | {r['air_time']*100:5.0f}% "
                     f"{r['lift_mm']:4.1f}mm {r['step_ratio']*100:6.0f}% | "
                     f"{r.get('sway_mm', 0):4.0f}mm {r.get('roll_deg', 0):4.1f}d {r.get('yaw_osc_deg', 0):5.1f}d")
    lines += [fmt(r) for r in good[:10]]
    calm = None
    if good:
        near = [r for r in good if r["dx"] >= 0.6 * good[0]["dx"]] or good[:1]
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


def refine(variant, t_long=32.0):
    """Around the best gait of walk.json: a finer local grid of the gait, then IMU
    feedback gains on the best few, then a long run (t_long s) of the winners; the
    best that still walks straight, within the torque limit and on real steps, and
    survives the long run, becomes walk.json's best (walk.txt gets a section)."""
    scene = variants.scene(variant)
    path = os.path.join(variants.folder(variant), "walk.json")
    W = json.load(open(path))
    b = W["best"]
    keys = ("freq", "roll", "swing", "phase", "pair", "follow", "yaw", "fb")
    base = {k: b[k] for k in keys if k in b}
    span = lambda v, ds, lo=None: sorted({round(max(lo, v + d) if lo is not None else v + d, 3) for d in ds})
    gaits = []
    for f in span(b["freq"], (-0.2, -0.1, 0, 0.1, 0.2), 0.3):
        for r in span(b["roll"], (-1, 0, 1), 1):
            for sw in span(b["swing"], (-3, -1.5, 0, 1.5, 3), 0):
                for ph in span(b["phase"], (-30, -15, 0, 15, 30)):
                    for fo in ([b.get("follow")] if b.get("follow") is None else span(b["follow"], (-0.25, 0, 0.25), 0)):
                        gaits.append(dict(base, freq=f, roll=r, swing=sw, phase=ph % 360, follow=fo))
    t = time.time()
    with Pool(N_PROCS) as pool:
        res = pool.map(_job, [(scene, g) for g in gaits])
    ok = sorted((r for r in res if r["fell"] is None and r.get("hip_hit", 0) == 0 and abs(r["dyaw"]) < 20
                 and r["peak"] <= PEAK_MAX), key=lambda r: -r["dx"])
    top = [{k: r[k] for k in keys if k in r} for r in ok[:5]]
    fbs = [dict(ki=ki, kd=kd, kdr=kdr) for ki in (0.15, 0.3, 0.5) for kd in (0.1, 0.2, 0.35) for kdr in (0.0, 0.03, 0.06)]
    with Pool(N_PROCS) as pool:
        res2 = pool.map(_job, [(scene, dict(g, fb=fb)) for g in top for fb in fbs])
    ok2 = sorted((r for r in res2 + ok if r["fell"] is None and r.get("hip_hit", 0) == 0 and abs(r["dyaw"]) < 20
                  and r["peak"] <= PEAK_MAX), key=lambda r: -r["dx"])
    with Pool(N_PROCS) as pool:
        checked = pool.map(_stats_job, [(scene, r) for r in ok2[:20]])
    good = [r for r in checked if r["step_ratio"] >= STEP_RATIO_MIN]
    with Pool(N_PROCS) as pool:
        longs = pool.map(_long_job, [(scene, {k: r[k] for k in keys if k in r}, t_long) for r in good[:8]])
    final = None
    for r, lr in zip(good[:8], longs):
        r["long_run"] = dict(seconds=t_long, fell=lr["fell"], dx=lr["dx"], dyaw=lr["dyaw"], peak=lr["peak"])
        if final is None and lr["fell"] is None and abs(lr["dyaw"]) < 30:
            final = r
    lines = ["", f"refined around the best gait: {len(gaits)} local gaits + {len(top)} x {len(fbs)} feedback "
             f"gains ({time.time()-t:.0f} s); {len(good)} of the top 20 step for real; long runs of {t_long:.0f} s:"]
    for r in good[:8]:
        lr = r["long_run"]
        lines.append(f"  freq {r['freq']:.1f} roll {r['roll']} swing {r['swing']} phase {r['phase']} follow "
                     f"{r.get('follow')} fb {r.get('fb')} | {r['dx']:.0f} mm in 12 s, peak {r['peak']:.0f}%, "
                     f"stepped {r['step_ratio']*100:.0f}% | {t_long:.0f} s: "
                     + (f"fell at {lr['fell']} s" if lr["fell"] else f"{lr['dx']:.0f} mm, turned {lr['dyaw']:.0f} deg"))
    if final and final["dx"] >= (b["dx"] if b else -1e9):
        W["best"], W["refined"] = final, True
        lines.append(f"new best: {final['dx']:.0f} mm in 12 s (was {b['dx']:.0f})")
    else:
        lines.append("the search's best gait stays")
    W["refine_top"] = good[:8]
    json.dump(W, open(path, "w"), indent=2, default=float)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(variants.folder(variant), "walk.txt"), "a") as fh:
        fh.write(text + "\n")
    return final


def best_gait(variant):
    with open(os.path.join(variants.folder(variant), "walk.json")) as fh:
        return json.load(fh)["best"]


def main():
    ap = variants.parser(__doc__)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--fb", help="feedback balance gains ki,kd,kdr (e.g. 0.3,0.2,0.03), see balance.py")
    ap.add_argument("--refine", action="store_true", help="finer search around walk.json's best gait")
    ap.add_argument("--video")
    a = ap.parse_args()
    fb = None
    if a.fb:
        ki, kd, kdr = (float(x) for x in a.fb.split(","))
        fb = dict(ki=ki, kd=kd, kdr=kdr)
    if a.search:
        search(a.variant, fb)
    if a.refine:
        refine(a.variant)
    if a.video:
        import walk_video
        walk_video.render(a.variant, best_gait(a.variant), a.video)


if __name__ == "__main__":
    main()
