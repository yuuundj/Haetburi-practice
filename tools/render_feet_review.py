#!/usr/bin/env python3
"""Side-by-side review sheet: source STL (left) vs the version with feet (right).

    MUJOCO_GL=glfw python3 tools/render_feet_review.py  -> review/feet_v1.png
"""
import os
import tempfile

import mujoco
import trimesh
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "modelings", "haetburi_grip_20mm_300mm.stl")
NEW = os.path.join(ROOT, "modelings", "haetburi_feet_v1.stl")
OUT = os.path.join(ROOT, "review", "feet_v1.png")
W, H = 560, 560

XML = """
<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55"/></visual>
  <asset>
{MESHES}
    <texture type="2d" name="grid" builtin="checker" rgb1="0.86 0.87 0.89" rgb2="0.78 0.80 0.83"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12"/>
    <material name="blue" rgba="0.106 0.216 0.541 1" specular="0.35" shininess="0.5"/>
    <material name="ghost" rgba="0.55 0.60 0.70 1" specular="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.2" dir="-0.2 0.5 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.01" material="grid"/>
    <body name="src" pos="-0.11 0 0">{SRC_GEOMS}</body>
    <body name="new" pos="0.11 0 0">{NEW_GEOMS}</body>
  </worldbody>
</mujoco>"""


def split_mesh(path, name, tmp):
    """MuJoCo caps a mesh at 200k faces; hand it the model in two halves."""
    mesh = trimesh.load(path)
    half = len(mesh.faces) // 2
    assets, geoms = [], []
    for k, idx in enumerate((np.arange(half), np.arange(half, len(mesh.faces)))):
        f = os.path.join(tmp, f"{name}{k}.stl")
        mesh.submesh([idx], append=True).export(f)
        assets.append(f'    <mesh name="{name}{k}" file="{f}" scale="0.001 0.001 0.001"/>')
        mat = "ghost" if name == "src" else "blue"
        grp = 1 if name == "src" else 2
        geoms.append(f'<geom type="mesh" mesh="{name}{k}" material="{mat}" group="{grp}"/>')
    return assets, geoms


def main():
    tmp = tempfile.mkdtemp()
    sa, sg_ = split_mesh(SRC, "src", tmp)
    na, ng = split_mesh(NEW, "new", tmp)
    xml = XML.format(W=W, H=H, MESHES="\n".join(sa + na),
                     SRC_GEOMS="".join(sg_), NEW_GEOMS="".join(ng))
    path = os.path.join(tmp, "review.xml")
    with open(path, "w") as fh:
        fh.write(xml)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    floor_id, src_id, new_id = 0, 1, 2          # geom groups

    def shot(label, az, el, dist, look, hide=(), sub=None):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = look
        for grp in (0, 1, 2):
            opt.geomgroup[grp] = 0 if grp in hide else 1
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        g = ImageDraw.Draw(img)
        g.rectangle([0, 0, W, 26], fill=(18, 20, 28))
        g.text((8, 7), label, fill=(240, 240, 240))
        if sub:
            g.text((8, H - 20), sub, fill=(40, 40, 40))
        return np.array(img)

    front_note = "grey = source STL, blue = feet_v1"
    tiles = [
        shot("front: source (left) vs feet_v1 (right)", 90, -4, 0.95, [0, 0, 0.15], sub=front_note),
        shot("3/4 view", 55, -12, 0.95, [0, 0, 0.15], sub=front_note),
        shot("side: face to the right", 0, -4, 0.75, [0.11, 0, 0.15], hide=(src_id,)),
        shot("close-up front", 90, -8, 0.30, [0.11, 0, 0.03], hide=(src_id,)),
        shot("close-up 3/4 from the front", 60, -18, 0.30, [0.11, -0.005, 0.03], hide=(src_id,)),
        shot("close-up side", 0, -6, 0.30, [0.11, 0, 0.03], hide=(src_id,)),
        shot("close-up back 3/4", 235, -15, 0.30, [0.11, 0, 0.03], hide=(src_id,)),
        shot("source legs, same close-up", 60, -18, 0.30, [-0.11, -0.005, 0.03], hide=(new_id,)),
        shot("sole from below", 90, 70, 0.26, [0.11, -0.005, 0.0], hide=(src_id, floor_id)),
    ]
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Image.fromarray(sheet).save(OUT)
    print("wrote", OUT, sheet.shape)


if __name__ == "__main__":
    main()
