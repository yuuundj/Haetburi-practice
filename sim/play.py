#!/usr/bin/env python3
"""Pick a model and watch it walk in the MuJoCo viewer, live, tuning the gait as it goes.

    python3 sim/play.py                          # menu of the built models
    python3 sim/play.py --variant v14_sts3032_hidden
    python3 sim/play.py --variant v12_slipper_display --gait calm
    python3 sim/play.py --variant v9_yaw_roll_display --freq 1.1 --roll 4 --swing 20 --phase 75
    python3 sim/play.py --variant v14_scs0009 --headless --seconds 12     # no window: numbers only

The gait is the open-loop rocking gait of sim/walk.py: the ankle rolls rock the
robot onto one foot, the hips swing the other foot forward.  It starts from the
model's best gait in robot/<variant>/walk.json (or "calm", or your own numbers).

Keys in the viewer window (the robot keeps walking while you change them):
    Up / Down          frequency   +- 0.1 Hz
    Right / Left       hip swing   +- 2 deg
    Page Up / Down     ankle roll  +- 1 deg
    Insert             hips "same" <-> "anti" (Otto's gait / scissoring legs)
    Home               walk <-> stand still
    End                reset: robot back at the start, standing
Mouse: left drag orbits, right drag pans, scroll zooms; Tab opens MuJoCo's panels.
The camera follows the robot.  Every 2 s the terminal prints distance, heading,
side drift and the highest motor load (% of the servo's stall torque).
"""
import argparse
import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import balance
import variants
import walk

KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT = 265, 264, 262, 263
KEY_PGUP, KEY_PGDN, KEY_INSERT, KEY_HOME, KEY_END = 266, 267, 260, 268, 269


def gait_of(variant, which):
    path = os.path.join(variants.folder(variant), "walk.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        js = json.load(fh)
    g = js.get(which) or js.get("best")
    return None if g is None else {k: g[k] for k in ("freq", "roll", "swing", "phase", "pair", "follow", "fb") if k in g}


def menu():
    rows = []
    for v in variants.available():
        try:
            info = variants.info(v)
        except FileNotFoundError:
            continue
        g = gait_of(v, "best")
        rows.append((v, info.get("height_mm", 0), info.get("mass_kg", 0), info.get("title", ""), g))
    print("\nBuilt models (the ones with a gait were searched by sim/walk.py --search):\n")
    for i, (v, h, mkg, title, g) in enumerate(rows, 1):
        gs = f"{g['freq']} Hz, roll {g['roll']}, hip {g['swing']}" if g else "no gait yet (defaults)"
        print(f"  {i:2d}. {v:26s} {h:5.0f} mm {mkg:5.2f} kg   {gs}")
        print(f"      {title}")
    while True:
        s = input("\nnumber or name: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(rows):
            return rows[int(s) - 1][0]
        if s in [r[0] for r in rows]:
            return s
        print("not in the list")


class Walker:
    """Drives the oscillators and keeps the running statistics."""

    def __init__(self, m, d, g):
        self.m, self.d, self.g = m, d, dict(g)
        self.swing = walk.swing_joints(m)
        self.walking, self.t0 = True, 0.0
        self.act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
        self.stall = float(m.actuator_forcerange[:, 1].max())
        self.pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.fb = balance.make(self.m, self.g)          # feedback balance, if the gait has it
        self.t0 = self.d.time
        self.start = self.d.xpos[self.pel].copy()
        self.yaw0 = walk.yaw_of(self.d.xmat[self.pel])
        self.z0 = self.d.subtree_com[1][2]
        self.peak, self.fell, self.next_report = 0.0, None, 2.0

    def step(self):
        d = self.d
        t = d.time - self.t0
        tg = walk.targets(t, self.g, self.swing) if self.walking else {}
        if self.fb is not None:
            self.fb.update(d, tg, self.m.opt.timestep)
        d.ctrl[:] = 0.0
        for k, v in tg.items():
            if k in self.act:
                d.ctrl[self.act[k]] = v
        mujoco.mj_step(self.m, d)
        if t > walk.RAMP:
            self.peak = max(self.peak, float(np.abs(d.actuator_force).max()) / self.stall * 100)
        if self.fell is None and d.subtree_com[1][2] < 0.75 * self.z0:
            self.fell = t
            print(f"  ** fell over at {t:.1f} s  (End: reset)")

    def status(self):
        d, g = self.d, self.g
        p = d.xpos[self.pel] - self.start
        yaw = walk.yaw_of(d.xmat[self.pel]) - self.yaw0
        mode = "walking" if self.walking else "standing"
        return (f"t {d.time - self.t0:5.1f} s  {mode}  forward {p[0]*1000:+6.0f} mm  side {p[1]*1000:+5.0f} mm  "
                f"heading {yaw:+4.0f} deg  peak load {self.peak:3.0f} %   |  {g['freq']:.1f} Hz, roll {g['roll']:.0f}, "
                f"hip {g['swing']:.0f}, phase {g['phase']:.0f}, {g.get('pair', 'anti')}")

    def key(self, code):
        g = self.g
        if code == KEY_UP:
            g["freq"] = round(min(2.5, g["freq"] + 0.1), 2)
        elif code == KEY_DOWN:
            g["freq"] = round(max(0.2, g["freq"] - 0.1), 2)
        elif code == KEY_RIGHT:
            g["swing"] = min(40, g["swing"] + 2)
        elif code == KEY_LEFT:
            g["swing"] = max(0, g["swing"] - 2)
        elif code == KEY_PGUP:
            g["roll"] = min(15, g["roll"] + 1)
        elif code == KEY_PGDN:
            g["roll"] = max(0, g["roll"] - 1)
        elif code == KEY_INSERT:
            g["pair"] = "anti" if g.get("pair", "anti") == "same" else "same"
        elif code == KEY_HOME:
            self.walking = not self.walking
        elif code == KEY_END:
            self.reset()
        else:
            return False
        print("  " + self.status())
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", help="model folder under robot/ (menu if omitted)")
    ap.add_argument("--gait", default="best", choices=("best", "calm"), help="gait from walk.json")
    for k, h in (("freq", "Hz"), ("roll", "ankle roll amplitude, deg"), ("swing", "hip swing amplitude, deg"),
                 ("phase", "hip phase after the roll, deg")):
        ap.add_argument(f"--{k}", type=float, help=h)
    ap.add_argument("--pair", choices=("same", "anti"))
    ap.add_argument("--headless", action="store_true", help="no window: run and print the result")
    ap.add_argument("--seconds", type=float, default=12.0, help="headless run time")
    ap.add_argument("--quit-after", type=float, help=argparse.SUPPRESS)      # smoke tests
    a = ap.parse_args()

    variant = a.variant or menu()
    scene = variants.scene(variant)
    g = gait_of(variant, a.gait) or dict(freq=0.9, roll=4, swing=12, phase=90, pair="same")
    for k in ("freq", "roll", "swing", "phase", "pair"):
        if getattr(a, k) is not None:
            g[k] = getattr(a, k)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    w = Walker(m, d, g)
    print(f"\n{variant}: {variants.info(variant).get('title', '')}")
    print(f"servo stall torque {w.stall} N.m, {m.nu} motors, swing joints: {w.swing}")

    if a.headless:
        while d.time - w.t0 < a.seconds and w.fell is None:
            w.step()
        print(w.status() + ("" if w.fell is None else f"   (fell at {w.fell:.1f} s)"))
        return

    print(__doc__[__doc__.index("Keys"):])
    keys = []                       # the viewer calls back on its own thread: queue, apply in the loop
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as v:
        v.cam.distance = 3.0 * w.z0
        v.cam.azimuth, v.cam.elevation = 200, -12
        wall0, sim0 = time.time(), d.time
        if a.quit_after:                         # smoke test: press a few keys by itself
            keys.extend([KEY_UP, KEY_RIGHT, KEY_INSERT, KEY_INSERT])
        while v.is_running():
            with v.lock():
                while keys:
                    w.key(keys.pop(0))
                # keep up with the wall clock (real time), at most 50 ms of physics per frame
                target = sim0 + (time.time() - wall0)
                n = 0
                while d.time < target and n < 25:
                    w.step()
                    n += 1
                if d.time < target - 0.2:             # the computer cannot keep up: let it lag
                    wall0, sim0 = time.time(), d.time
                v.cam.lookat[:] = [d.xpos[w.pel][0], d.xpos[w.pel][1], 1.1 * w.z0]
            v.sync()
            if a.quit_after and d.time - w.t0 >= a.quit_after:
                break
            if d.time - w.t0 >= w.next_report:
                w.next_report += 2.0
                print("  " + w.status())
            time.sleep(0.005)


if __name__ == "__main__":
    sys.exit(main())
