#!/usr/bin/env python3
"""Hip seam review for the hip pitch variants (v17):  -> robot/<variant>/hip_seam.png

  row 1  the seam up close (legs tinted), standing and at the hip limits
  row 2  side section through the left leg at the limits, with the seam sphere,
         and the collision-free hip yaw x pitch map (tools/build_model.hip_range)

    MUJOCO_GL=egl python3 tools/render_hip_seam.py v17_sts3032
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 520, 520


def main(v):
    base = os.path.join(ROOT, "robot", v)
    info = json.load(open(os.path.join(base, "variant.json")))
    rng = json.load(open(os.path.join(base, "hip_range.json")))
    mR = re.search(r"sphere R ([\d.]+) mm", info["summary"]["hip seam"])
    R = float(mR.group(1)) / 1000 if mR else None          # v18: no sphere, the belly is carved
    m = mujoco.MjModel.from_xml_path(os.path.join(base, "haetburi_scene.xml"))
    d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    lo, hi = np.degrees(m.jnt_range[jid("hip_pitch_l")])
    ylim = np.degrees(m.jnt_range[jid("hip_yaw_l")][1])

    def setpose(q):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for j, deg in q.items():
            d.qpos[m.jnt_qposadr[jid(j)]] = np.radians(deg)
        mujoco.mj_forward(m, d)

    # ---- renders: legs tinted so the seam shows
    leg_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (bid("leg_l"), bid("leg_r"))
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] != 3]
    for g in leg_geoms:
        m.geom_matid[g] = -1
        m.geom_rgba[g] = [0.95, 0.55, 0.15, 1]
    for mt in ("display", "screen"):
        m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, mt)][3] = 0.0
    r = mujoco.Renderer(m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    setpose({})
    look = d.xpos[bid("hip_l")].copy() + [0, 0, -0.012]

    def shot(label, az, el, dist, at=look):
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = at
        r.update_scene(d, cam, opt)
        im = Image.fromarray(r.render())
        g = ImageDraw.Draw(im)
        g.rectangle([0, 0, W, 24], fill=(18, 20, 28))
        g.text((8, 7), label, fill=(240, 240, 240))
        return np.array(im)

    D = 0.16 * info["height_mm"] / 400
    tiles = []
    setpose({})
    tiles.append(shot(f"{v}: standing (leg = orange, display hidden)", 120, -8, D))
    setpose({"hip_pitch_l": lo})
    tiles.append(shot(f"left leg forward {-lo:.0f} deg (limit)", 90, -5, D * 1.15))
    setpose({"hip_pitch_l": hi})
    tiles.append(shot(f"left leg back {hi:.0f} deg (limit)", 90, -5, D * 1.15))
    setpose({"hip_yaw_l": ylim, "hip_yaw_r": -ylim})
    tiles.append(shot(f"hip yaw {ylim:.0f} deg, toes out", 180, 35, D))
    setpose({})
    hide = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("leg_", "foot_", "hip_"))
            or m.geom_bodyid[g] == 0]                          # the floor too: look up at the belly
    keep = m.geom_rgba[hide].copy()
    m.geom_rgba[hide, 3] = 0.0
    mats = m.geom_matid[hide].copy()
    m.geom_matid[hide] = -1
    tiles.append(shot("lower belly from below (legs, feet, floor hidden)", 160, 50, D * 0.85))
    m.geom_rgba[hide], m.geom_matid[hide] = keep, mats
    row1 = np.hstack(tiles)

    # ---- side section through the left leg
    def world_mesh(names):
        out = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] == 3:
                continue
            if m.body(m.geom_bodyid[g]).name not in names:
                continue
            mid = m.geom_dataid[g]
            v_ = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            f_ = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            out.append(trimesh.Trimesh(v_ @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f_, process=False))
        return trimesh.util.concatenate(out)

    def section(mesh, y):
        s = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        return [] if s is None else [l[:, [0, 2]] for l in s.discrete]

    setpose({})
    hip = d.xpos[bid("hip_l")].copy()
    piv = d.xanchor[jid("hip_pitch_l")].copy()
    y = hip[1]
    rel = lambda p: (p - [piv[0], piv[2]]) * 1000
    fig = plt.figure(figsize=(W * 2 / 100, H / 100), dpi=100)
    ax = fig.add_subplot(1, 2, 1)
    for l in section(world_mesh({"pelvis", "torso"}), y):
        ax.plot(*rel(l).T, color="#1b378a", lw=1.6)
    for q, col, lab in ((0, "#e08a1e", "standing"), (lo, "#2a9d4b", f"forward {-lo:.0f}"),
                        (hi, "#c0392b", f"back {hi:.0f}")):
        setpose({"hip_pitch_l": q})
        first = True
        for l in section(world_mesh({"leg_l"}), y):
            ax.plot(*rel(l).T, color=col, lw=1.4, label=lab if first else None)
            first = False
    setpose({})
    t = np.linspace(0, 2 * np.pi, 300)
    if R is not None:
        ax.plot(R * 1000 * np.cos(t), R * 1000 * np.sin(t), "--", color="#777", lw=0.9,
                label=f"seam sphere R {R*1000:.1f}")
    ax.axhline((hip[2] - piv[2]) * 1000, color="#999", lw=0.8, ls=":", label="original crotch line")
    for g in range(m.ngeom):                       # hip motors (boxes) in this plane
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.body(m.geom_bodyid[g]).name in ("pelvis", "hip_l") \
                and m.geom_matid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "motor"):
            c, s_ = d.geom_xpos[g], m.geom_size[g]
            if abs(c[1] - y) < s_[1]:
                x0, z0 = rel(np.array([c[0] - s_[0], c[2] - s_[2]]))
                ax.add_patch(plt.Rectangle((x0, z0), 2 * s_[0] * 1000, 2 * s_[2] * 1000, color="#c22", alpha=0.35))
    ax.plot(0, 0, "k+", ms=10)
    ax.set_aspect("equal")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 30)
    ax.invert_xaxis()                              # robot front (+x) to the left, as in the renders
    ax.set_xlabel("mm, front <-  -> back (from the hip pitch axis)")
    ax.set_title("side section through the left leg", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    hits = np.array([[len(h) == 0 for h in row] for row in rng["hits"]], float)
    ax2.imshow(hits, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[rng["pitch"][0] - 1, rng["pitch"][-1] + 1, rng["yaw"][0] - 2.5, rng["yaw"][-1] + 2.5])
    ax2.add_patch(plt.Rectangle((lo, -ylim), hi - lo, 2 * ylim, fill=False, ec="k", lw=1.5, label="joint limits set"))
    ax2.set_xlabel("left hip pitch (deg)   - forward  |  + back")
    ax2.set_ylabel("left hip yaw (deg)   - toes in | + toes out")
    ax2.set_title("green: no collision (leg+foot vs belly, backpack, other leg/foot, display)", fontsize=9)
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.canvas.draw()
    row2 = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    row2 = np.array(Image.fromarray(row2).resize((row1.shape[1], int(row2.shape[0] * row1.shape[1] / row2.shape[1]))))
    out = os.path.join(base, "hip_seam.png")
    Image.fromarray(np.vstack([row1, row2])).save(out)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v17_sts3032")
