#!/usr/bin/env python3
"""Render robot/preview.png - a contact sheet of the finished model, including
x-ray views (shells made translucent) that show the motors inside.

    MUJOCO_GL=glfw python3 tools/render_preview.py
"""
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "robot", "haetburi_scene.xml")
OUT = os.path.join(ROOT, "robot", "preview.png")
W, H = 520, 660


def robot_height(m, d):
    top = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            top = max(top, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].max())
    return top


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
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

    def pose(targets, seconds=3.0):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for k, v in targets.items():
            d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)] = np.deg2rad(v)
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)

    tiles = []
    mode("solid")
    tiles.append(shot(f"front ({height*1000:.0f} mm)", 180, -6, 2.2, 0.5))
    tiles.append(shot("side (backpack behind)", 90, -6, 2.2, 0.5))
    tiles.append(shot("3/4", 215, -14, 2.2, 0.5))
    tiles.append(shot("legs, feet and ankle gap", 205, -8, 0.75, 0.17))
    mode("xray")
    tiles.append(shot("x-ray: roll motor in the leg, pitch motor in the foot", 205, -8, 0.75, 0.17))
    tiles.append(shot("x-ray: all 7 XL430 inside the shell", 215, -10, 2.0, 0.45))
    mode("solid")
    pose({"hip_pitch_l": 12, "hip_pitch_r": 12, "ankle_pitch_l": -9, "ankle_pitch_r": -9,
          "waist_yaw": -30})
    tiles.append(shot("hip bow + waist twist", 215, -10, 2.2, 0.5))
    pose({"ankle_roll_l": 6.5, "ankle_roll_r": 6.5})
    pose_t = {"ankle_roll_l": 6.5, "ankle_roll_r": 6.5, "hip_pitch_r": 15, "ankle_pitch_r": -15}
    for k, v in pose_t.items():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k)] = np.deg2rad(v)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    tiles.append(shot("ankle roll 6.5 + right hip 15 (both feet stay down)", 160, -6, 2.2, 0.5))
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mode("col")
    tiles.append(shot("collision geometry", 200, -10, 2.2, 0.5))
    m.mat_rgba[shell] = solid

    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
    Image.fromarray(sheet).save(OUT)
    print("wrote", OUT, sheet.shape)


if __name__ == "__main__":
    main()
