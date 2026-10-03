#!/usr/bin/env python3
"""Render a variant in the same three views as a reference sheet (front, side,
back) and stack it under the reference image.

    MUJOCO_GL=glfw python3 tools/render_compare.py v9_yaw_roll_display images-modeling/Haetburi_display.png
    -> robot/<variant>/reference_compare.png
"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(variant, ref_path, extra_views=True):
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "robot", variant, "haetburi_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ref = Image.open(os.path.join(ROOT, ref_path)).convert("RGB")
    W = ref.width // 3
    H = ref.height
    r = mujoco.Renderer(m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = [0, 0, 1, 0, 0, 0]                  # model only, no floor: like the sheet
    pel = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]

    def shot(az, el=0.0, dist=1.25, z=0.28):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [pel[0], pel[1], z]
        r.update_scene(d, cam, opt)
        return r.render()

    # reference sheet order: front, side (facing left), back
    views = [shot(180), shot(270), shot(0)]
    row = Image.fromarray(np.hstack(views))
    rows = [ref.resize((W * 3, H)), row]
    if extra_views:
        rows.append(Image.fromarray(np.hstack([shot(215, -12, 1.25), shot(160, -25, 0.75, 0.2),
                                               shot(125, -10, 0.75, 0.2)])))
    sheet = Image.new("RGB", (W * 3, sum(im.height for im in rows) + 26 * len(rows)), (18, 20, 28))
    y = 0
    labels = ["reference: " + os.path.basename(ref_path),
              f"{variant}: front / side / back",
              f"{variant}: 3/4, display close-up from the front and the side"]
    g = ImageDraw.Draw(sheet)
    for im, lab in zip(rows, labels):
        g.text((8, y + 7), lab, fill=(240, 240, 240))
        sheet.paste(im, (0, y + 26))
        y += im.height + 26
    out = os.path.join(ROOT, "robot", variant, "reference_compare.png")
    sheet.save(out)
    print("wrote", os.path.relpath(out, ROOT), sheet.size)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
