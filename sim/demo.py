#!/usr/bin/env python3
"""Exercise every joint in turn so the range of motion is visible.

    python3 sim/demo.py [--variant v6_split_round]   # interactive viewer
    python3 sim/demo.py --video out.mp4
    python3 sim/demo.py --frames dir                 # numbered PNGs

Motion amplitudes come from robot/<variant>/variant.json (lean angles are kept
below each variant's tip-over limit).
"""
import argparse
import os

import mujoco
import numpy as np

import variants

SCENE = None                      # set in main() / configure()
SCRIPT, DURATION = [], 0.0


def configure(variant=variants.DEFAULT):
    """Build the motion script from the amplitudes stored with the variant."""
    global SCENE, SCRIPT, DURATION
    SCENE = variants.scene(variant)
    a = variants.info(variant)["demo"]
    SCRIPT = [
        (0.0, 1.5, "settle", {}),
        (1.5, 5.5, "ankle pitch: lean fore and aft",
         {"ankle_pitch_l": a["pitch_lean"], "ankle_pitch_r": a["pitch_lean"]}),
        (5.5, 9.5, "ankle roll: lean left and right",
         {"ankle_roll_l": a["roll_lean"], "ankle_roll_r": a["roll_lean"]}),
        (9.5, 13.5, "waist yaw: twist", {"waist_yaw": a["yaw"]}),
        (13.5, 18.0, "hip pitch + ankle pitch: bow",
         {"hip_pitch_l": a["bow_hip"], "hip_pitch_r": a["bow_hip"],
          "ankle_pitch_l": a["bow_ankle"], "ankle_pitch_r": a["bow_ankle"]}),
        (18.0, 22.0, "diagonal: roll one way, twist the other",
         {"ankle_roll_l": a["diag_roll"], "ankle_roll_r": a["diag_roll"],
          "waist_yaw": a["diag_yaw"]}),
        (22.0, 23.5, "settle", {}),
    ]
    DURATION = SCRIPT[-1][1]
    return SCENE


def targets_at(t):
    out = {}
    for t0, t1, _, amps in SCRIPT:
        if t0 <= t < t1:
            phase = (t - t0) / (t1 - t0)
            s = np.sin(2 * np.pi * phase)
            for k, a in amps.items():
                out[k] = np.deg2rad(a) * s
    return out


def label_at(t):
    for t0, t1, lab, _ in SCRIPT:
        if t0 <= t < t1:
            return lab
    return ""


def drive(m, d, t):
    for k, v in targets_at(t).items():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)] = v


def run_viewer(m, d):
    import time
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0, 0, 0.35]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -12
        last = ""
        while v.is_running():
            t0 = time.time()
            t = d.time % DURATION
            drive(m, d, t)
            mujoco.mj_step(m, d)
            if label_at(t) != last:
                last = label_at(t)
                print(f"  {d.time:6.1f} s  {last}")
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))


def run_offscreen(m, d, video, frames_dir, fps=30):
    from PIL import Image
    r = mujoco.Renderer(m, 720, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.35]
    cam.distance, cam.azimuth, cam.elevation = 1.55, 200, -12
    step = int(round(1.0 / fps / m.opt.timestep))
    frames = []
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    n = 0
    while d.time < DURATION:
        drive(m, d, d.time)
        for _ in range(step):
            mujoco.mj_step(m, d)
        cam.azimuth = 200 + 25 * np.sin(2 * np.pi * d.time / DURATION)
        r.update_scene(d, cam)
        img = r.render()
        if frames_dir:
            Image.fromarray(img).save(os.path.join(frames_dir, f"{n:04d}.png"))
        frames.append(img)
        n += 1
    if video:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=fps)
        print("wrote", video)
    if frames_dir:
        print(f"wrote {n} frames to {frames_dir}")


def main():
    ap = variants.parser(__doc__)
    ap.add_argument("--video")
    ap.add_argument("--frames")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(configure(a.variant))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if a.video or a.frames:
        run_offscreen(m, d, a.video, a.frames)
    else:
        run_viewer(m, d)


if __name__ == "__main__":
    main()
