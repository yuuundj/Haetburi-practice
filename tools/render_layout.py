#!/usr/bin/env python3
"""Where everything sits inside a variant: x-ray views (front, side, top, 3/4) with
each electronics part in its own colour and number, the servos in red, the display,
and the centre of mass, plus a legend.   -> robot/<variant>/layout_xray.png

    MUJOCO_GL=egl python3 tools/render_layout.py v22_sts3032
"""
import json
import os
import re
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 640, 640
COLOURS = [(0.95, 0.75, 0.10), (0.10, 0.75, 0.30), (0.20, 0.55, 1.00), (0.85, 0.30, 0.95),
           (0.10, 0.85, 0.90), (1.00, 0.50, 0.20)]


def font(size):
    for f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main(v):
    base = os.path.join(ROOT, "robot", v)
    parts = [p for p in json.load(open(os.path.join(base, "electronics.json"))) if p.get("size_mm")
             and not p.get("unplaced")]
    rep = open(os.path.join(base, "build_report.txt")).read()
    k = float(re.search(r"character scale x([\d.]+)", rep).group(1))
    m = mujoco.MjModel.from_xml_path(os.path.join(base, "haetburi_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mat = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n)
    shell, motor, foot, disp, scr = mat("shell"), mat("motor"), mat("foot"), mat("display"), mat("screen")
    m.mat_rgba[shell] = [0.35, 0.45, 0.75, 0.13]
    m.mat_rgba[disp] = [0.15, 0.15, 0.15, 0.55]
    m.mat_rgba[scr] = [0.6, 0.8, 1.0, 0.25]
    m.mat_rgba[motor] = [0.90, 0.10, 0.10, 1.0]
    # the electronics boxes: pelvis visual boxes in the "foot" material, matched to the parts by position
    boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pel and m.geom_matid[g] == foot
             and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    want = [np.array([-p["behind"] * k, p["side"] * k, p["up"] * k]) / 1000.0 for p in parts]   # pelvis frame
    found = {}
    for i, w in enumerate(want):
        g = min(boxes, key=lambda g: np.linalg.norm(m.geom_pos[g] - w))
        found[i] = g
        m.geom_matid[g] = -1
        m.geom_rgba[g] = list(COLOURS[i % len(COLOURS)]) + [1.0]
    for g in range(m.ngeom):                       # the floor out of the way
        if m.geom_bodyid[g] == 0:
            m.geom_rgba[g, 3] = 0.0
            m.geom_matid[g] = -1
    com = d.subtree_com[pel].copy()
    top = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            top = max(top, d.geom_xpos[g][2] + 0.01)
    height = max(top, 0.3)

    r = mujoco.Renderer(m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    fovy = np.radians(m.vis.global_.fovy)

    def project(p, cam):
        a, e = np.radians(cam.azimuth), np.radians(cam.elevation)
        f = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        pos = np.array(cam.lookat) - cam.distance * f
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        q = p - pos
        z = q @ f
        s = (H / 2) / np.tan(fovy / 2)
        return W / 2 + (q @ right) / z * s, H / 2 - (q @ up) / z * s

    F, Fs = font(20), font(15)
    look = np.array([d.xpos[pel][0], 0.0, height * 0.42])
    centre = np.mean([d.geom_xpos[g] for g in found.values()] + [com], axis=0)    # the parts and the CoM
    tiles = []
    views = (("whole robot, front", 180, -2, 1.55 * height, look),
             ("whole robot, side (front to the left)", 90, -2, 1.55 * height, look),
             ("close-up, side (front to the left)", 90, -2, 0.75 * height, centre),
             ("close-up, from behind", 0, -2, 0.75 * height, centre),
             ("close-up, from above", 180, -89, 0.75 * height, centre),
             ("close-up, 3/4 from behind", 35, -20, 0.75 * height, centre))
    for label, az, el, dist, at in views:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = at
        r.update_scene(d, cam, opt)
        im = Image.fromarray(r.render())
        g = ImageDraw.Draw(im)
        for i, gid in found.items():                # numbers at the parts
            u, vv = project(d.geom_xpos[gid], cam)
            g.ellipse([u - 11, vv - 11, u + 11, vv + 11], fill=(255, 255, 255), outline=(0, 0, 0))
            g.text((u - 5, vv - 10), str(i + 1), fill=(0, 0, 0), font=Fs)
        u, vv = project(com, cam)                   # centre of mass
        g.ellipse([u - 9, vv - 9, u + 9, vv + 9], outline=(0, 0, 0), width=3)
        g.line([u - 14, vv, u + 14, vv], fill=(0, 0, 0), width=2)
        g.line([u, vv - 14, u, vv + 14], fill=(0, 0, 0), width=2)
        g.rectangle([0, 0, W, 30], fill=(18, 20, 28))
        g.text((8, 5), f"{v}: {label}", fill=(240, 240, 240), font=F)
        tiles.append(np.array(im))
    sheet = Image.fromarray(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    # legend
    crotch = re.search(r"legs raised to clear the feet \(mm\): ([\d.]+)", rep)
    lift = float(crotch.group(1)) if crotch else 0.0
    z_crotch = 9.8 * k + lift
    mass = re.search(r"total mass ([\d.]+) kg", rep).group(1)
    leg = Image.new("RGB", (sheet.width, 300), (250, 250, 250))
    g = ImageDraw.Draw(leg)
    g.text((14, 10), f"{v}: {mass} kg.  Red: servos.  Grey: display.  Crosshair: centre of mass "
                     f"({com[0]*1000 - d.xpos[pel][0]*1000:+.0f} mm ahead of the leg-centre line, "
                     f"{com[2]*1000:.0f} mm above the floor).   Positions: behind the leg-centre line / "
                     f"above the floor / left of the middle (mm)", fill=(0, 0, 0), font=Fs)
    y = 44
    for i, p in enumerate(parts):
        c = tuple(int(255 * x) for x in COLOURS[i % len(COLOURS)])
        g.rectangle([14, y, 40, y + 22], fill=c, outline=(0, 0, 0))
        g.text((22, y + 2), str(i + 1), fill=(0, 0, 0), font=Fs)
        g.text((52, y + 1), f"{p['name']}  {p['kg']*1000:.0f} g  ({' x '.join(f'{s:.0f}' for s in p['size_mm'])} mm)"
                            f"   ->   {p['behind']*k:+.0f} / {z_crotch + p['up']*k:.0f} / {p['side']*k:+.0f}",
               fill=(0, 0, 0), font=F)
        y += 34
    out = os.path.join(base, "layout_xray.png")
    Image.fromarray(np.vstack([np.array(sheet), np.array(leg)])).save(out)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1])
