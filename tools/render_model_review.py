#!/usr/bin/env python3
"""Review sheet for the new base model: the reference sheet, the model in the
same four views (front, left side, right side, back), and close-ups of the tail
and of the hands holding the display.   -> review/display_model_v2.png

    MUJOCO_GL=glfw python3 tools/render_model_review.py
"""
import os
import sys
import tempfile

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_display_model as MD  # noqa: E402

ROOT = MD.ROOT
REF = os.path.join(ROOT, "images-modeling", "Haetburi_display.png")
OUT = os.path.join(ROOT, "review", "display_model_v2.png")


def split_assets(mesh, name, tmp, mat, scale):
    """MuJoCo caps a mesh at 200k faces: hand it over in chunks."""
    assets, geoms = [], []
    chunks = np.array_split(np.arange(len(mesh.faces)), len(mesh.faces) // 150000 + 1)
    for k, idx in enumerate(chunks):
        f = os.path.join(tmp, f"{name}{k}.stl")
        mesh.submesh([idx], append=True).export(f)
        assets.append(f'<mesh name="{name}{k}" file="{f}" scale="{scale} {scale} {scale}"/>')
        geoms.append(f'<geom type="mesh" mesh="{name}{k}" material="{mat}"/>')
    return assets, geoms


def main():
    parts = MD.main_parts(trimesh.load(MD.NEW))
    box, c, (w, h, t), _ = MD.display_box(parts)
    character = trimesh.load(MD.OUT_BASE)                    # arms widened, tail on
    s = MD.CHAR_HEIGHT_MM / MD.NEW_HEIGHT / 1000.0            # native -> metres, 500 mm character
    tmp = tempfile.mkdtemp()
    a1, g1 = split_assets(character, "body", tmp, "blue", s)
    sc = (np.array(c) + [0, -t / 2 - 0.02, 0]) * s
    xml = f"""<mujoco><visual><global offwidth="1600" offheight="1200"/><quality offsamples="8"/>
      <headlight ambient="0.4 0.4 0.4" diffuse="0.55 0.55 0.55"/></visual>
      <asset>{''.join(a1)}
        <material name="blue" rgba="0.106 0.216 0.541 1" specular="0.3"/>
        <material name="disp" rgba="0.18 0.19 0.21 1" specular="0.6"/>
        <material name="screen" rgba="0.62 0.80 0.95 1" emission="0.25"/></asset>
      <worldbody><light pos="0.4 -0.8 1.2" dir="-0.3 0.6 -1" directional="true"/>
        <light pos="-0.5 0.8 1.0" dir="0.4 -0.6 -1" directional="true" diffuse="0.3 0.3 0.3"/>
        {''.join(g1)}
        <geom type="box" size="{w*s/2} {t*s/2} {h*s/2}" pos="{c[0]*s} {c[1]*s} {c[2]*s}" material="disp"/>
        <geom type="box" size="{192/210.5*w*s/2} 0.0002 {120/142.3*h*s/2}" pos="{sc[0]} {sc[1]} {sc[2]}" material="screen"/>
      </worldbody></mujoco>"""
    path = os.path.join(tmp, "m.xml")
    open(path, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ref = Image.open(REF).convert("RGB")
    W, H = ref.width // 4, ref.height
    cam = mujoco.MjvCamera()

    def shot(r, az, el, dist, look):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = look
        r.update_scene(d, cam)
        return r.render()

    r1 = mujoco.Renderer(m, H, W)
    mid = [0, 0, 0.25]
    # this model faces -y: front = camera on -y (azimuth 90), back = +y (270);
    # facing left in the picture = camera on -x (azimuth 0); facing right = +x (180)
    row1 = np.hstack([shot(r1, 90, 0, 1.25, mid), shot(r1, 0, 0, 1.25, mid),
                      shot(r1, 180, 0, 1.25, mid), shot(r1, 270, 0, 1.25, mid)])
    r2 = mujoco.Renderer(m, 520, W * 4 // 3)
    row2 = np.hstack([shot(r2, 225, -12, 0.5, [0, 0.05, 0.08]),          # tail from behind-left
                      shot(r2, 60, -6, 0.55, [0, -0.09, 0.14]),           # hands and display
                      shot(r2, 120, -14, 1.2, mid)])                       # 3/4
    rows = [(ref.resize((W * 4, H)), "reference: images-modeling/Haetburi_display.png"),
            (Image.fromarray(row1), "new base model: front / facing left / facing right / back "
                                    "(fists on the display ends, tail added)"),
            (Image.fromarray(row2), "tail from behind / hands and display / 3/4")]
    sheet = Image.new("RGB", (W * 4, sum(im.height + 26 for im, _ in rows)), (18, 20, 28))
    g = ImageDraw.Draw(sheet)
    y = 0
    for im, lab in rows:
        g.text((8, y + 7), lab, fill=(240, 240, 240))
        sheet.paste(im, (0, y + 26))
        y += im.height + 26
    sheet.save(OUT)
    print("wrote", os.path.relpath(OUT, ROOT), sheet.size)


if __name__ == "__main__":
    main()
