#!/usr/bin/env python3
"""Render robot/<variant>/preview.png - a contact sheet of one model, including
x-ray views (shells made translucent) that show where the motors sit.

    MUJOCO_GL=glfw python3 tools/render_preview.py [variant]
"""
import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 520, 660


def robot_height(m, d):
    top = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            top = max(top, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].max())
    return top


def render(variant):
    base = os.path.join(ROOT, "robot", variant)
    info = json.load(open(os.path.join(base, "variant.json")))
    demo = info["demo"]
    m = mujoco.MjModel.from_xml_path(os.path.join(base, "haetburi_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    height = robot_height(m, d)
    r = mujoco.Renderer(m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    shell = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "shell")
    solid = m.mat_rgba[shell].copy()

    def mode(kind):
        opt.geomgroup[:] = [1, 0, 1, 0, 0, 0] if kind != "col" else [1, 0, 0, 1, 0, 0]
        m.mat_rgba[shell] = solid if kind != "xray" else np.r_[solid[:3], 0.22]

    def shot(label, az, el, dist, look_z, look_x=0.0):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist * height
        cam.lookat[:] = [look_x, 0, look_z * height]
        r.update_scene(d, cam, opt)
        im = Image.fromarray(r.render())
        g = ImageDraw.Draw(im)
        g.rectangle([0, 0, W, 24], fill=(18, 20, 28))
        g.text((8, 7), label, fill=(240, 240, 240))
        return np.array(im)

    def pose(targets, seconds=3.0, ramp=1.5, reset=True):
        """Ease into the pose over `ramp` s, then hold; a step command would
        shove the lighter variants over."""
        if reset:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        start = d.ctrl.copy()
        goal = start.copy()
        for k, v in targets.items():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
            if i >= 0:
                goal[i] = np.deg2rad(v)
        n = int(ramp / m.opt.timestep)
        for step in range(int(seconds / m.opt.timestep)):
            a = min(1.0, (step + 1) / n)
            d.ctrl[:] = start + a * (goal - start)
            mujoco.mj_step(m, d)

    tiles = []
    mode("solid")
    tiles.append(shot(f"{variant}: front ({height*1000:.0f} mm)", 180, -6, 2.2, 0.5))
    tiles.append(shot("side (backpack behind)", 90, -6, 2.2, 0.5))
    tiles.append(shot("3/4", 215, -14, 2.2, 0.5))
    tiles.append(shot("legs, feet and ankle gap", 205, -8, 0.75, 0.17))
    mode("xray")
    tiles.append(shot("x-ray: legs and feet", 205, -8, 0.75, 0.17))
    servo = info["summary"].get("servo", "XL430-W250-T").split(":")[0]
    tiles.append(shot(f"x-ray: all {m.nu} {servo}", 215, -10, 2.0, 0.45))
    mode("solid")
    pose({"hip_pitch_l": demo["pose_bow_hip"], "hip_pitch_r": demo["pose_bow_hip"],
          "ankle_pitch_l": demo["pose_bow_ankle"], "ankle_pitch_r": demo["pose_bow_ankle"],
          "waist_yaw": -30})
    tiles.append(shot("hip bow + waist twist", 215, -10, 2.2, 0.5))
    lr, lh = demo["pose_lift_roll"], demo["pose_lift_hip"]
    pose({"ankle_roll_l": lr, "ankle_roll_r": lr})
    hipj = "hip_yaw" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_yaw_r") >= 0 else "hip_pitch"
    pose({"ankle_roll_l": lr, "ankle_roll_r": lr, f"{hipj}_r": lh,
          "ankle_pitch_r": -lh if hipj == "hip_pitch" else 0},
         seconds=1.5, ramp=1.0, reset=False)
    tiles.append(shot(f"ankle roll {lr} + right {hipj.replace('_', ' ')} {lh}", 160, -6, 2.2, 0.5))
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mode("col")
    tiles.append(shot("collision geometry", 200, -10, 2.2, 0.5))
    m.mat_rgba[shell] = solid

    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
    out = os.path.join(base, "preview.png")
    Image.fromarray(sheet).save(out)
    print("wrote", os.path.relpath(out, ROOT), sheet.shape)
    return tiles


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "v6_split_round")
