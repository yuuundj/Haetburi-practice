#!/usr/bin/env python3
"""Place the electronics of a hip-pitch variant (v19 / v20) where they fit and pull the
centre of mass down and back (toward the pitch axes, which sit behind it).

Each part is a box of its real size.  It must lie inside the body or the backpack, at
least one printed wall (WALL_MM) inside the skin and above the torso bottom.  It must
also stay clear of the hip pitch motor cases, of the round cavities the leg tops turn
in, and of the parts already placed.  The parts go in heaviest first; each takes the
spot that lowers  height + forward offset  of the CoM most (both in mm, equal weight).

    python3 tools/place_electronics.py v20_sts3032     -> robot/<variant>/electronics.json

The result is a list of (name, kg, behind, up, side, size_mm) in native units from the
leg centre at the crotch (+ behind, + up, + left).  The variant picks it up with
SMALL["elec"] = "placed"; tools/build_model.py draws the boxes.
"""
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# name, kg, size mm (the battery depends on the servo voltage, see PARTS_FOR)
PARTS_6V = [("battery 2S 7.4 V 1000 mAh LiPo", 0.060, (56, 31, 16)),
            ("servo driver with ESP32 + IMU", 0.020, (65, 30, 12)),
            ("Raspberry Pi Zero 2 W", 0.012, (65, 30, 6)),
            ("UBEC 6 V 5 A (servo power)", 0.012, (35, 18, 8)),
            ("buck 5 V 3 A (Pi, display)", 0.005, (22, 17, 6))]
PARTS_12V = [("battery 3S 11.1 V 1000 mAh LiPo", 0.085, (70, 35, 20))] + PARTS_6V[1:3] + \
            [("buck 5 V 3 A (Pi, display)", 0.005, (22, 17, 6))]
WIRING = ("wiring and connectors", 0.015)


def main(v):
    os.environ["HAETBURI_VARIANT"] = v
    import haetburi_params as P
    import newbase as NB
    k = P.MESH_SCALE
    parts = PARTS_12V if P.MOTOR_STALL_NM >= 0.9 else PARTS_6V
    rep = open(os.path.join(ROOT, "robot", v, "build_report.txt")).read()
    cut = re.search(r"torso cut flat ([\d.]+) mm", rep)
    socket = cut is not None                    # v19-v22: hip pitch in sockets; else the STL crotch
    zs = P.CUT_LEG_TOP + (float(cut.group(1)) / k if socket else 0.0)
    if socket:
        rdome = float(re.search(r"hidden dome R ([\d.]+) mm", rep).group(1)) / k
        hpm = re.search(r"axis ([\d.]+) mm above the crotch, ([\d.-]+) ahead of the leg centre .*?case ([+-][\d.]+) "
                        r"outboard", rep)
        ax_up, ax_ahead, ax_out = (float(hpm.group(i)) for i in (1, 2, 3))
    yaw_motor = getattr(P, "HIP_YAW", True) and "hip yaw" in rep

    surf = NB.part_surfaces()
    nb = os.path.join(ROOT, "robot", v, "body_native.stl")       # the build's re-shaped body (v21)
    if os.path.exists(nb):
        import trimesh as _tm
        surf["body_legs"] = _tm.load(nb)
    body, pack = NB.solid(surf["body_legs"]), NB.solid(surf["backpack"])
    if "--body-only" in sys.argv:                                 # nothing in the backpack
        pack = NB.Solid(np.zeros_like(pack.grid), pack.lo, pack.pitch)
    # the leg centres (as the build finds them) and the hip pitch cases / leg-top cavities
    import trimesh
    pts, _ = trimesh.sample.sample_surface(surf["body_legs"], 400000, seed=2)
    centres, keep_out = {}, []
    lo_c, hi_c = NB.case_extent(P)
    for sd, sgn in (("l", 1), ("r", -1)):
        band = pts[(pts[:, 2] < P.CUT_LEG_TOP) & (pts[:, 2] > P.CUT_LEG_TOP - 1.5) & (sgn * pts[:, 0] > 0)]
        cx, cy = (band[:, 0].min() + band[:, 0].max()) / 2, (band[:, 1].min() + band[:, 1].max()) / 2
        centres[sd] = (cx, cy)
        if socket:
            R = np.array([[-sgn, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
            shaft = np.array([cx + sgn * ax_out / k, cy - ax_ahead / k, P.CUT_LEG_TOP + ax_up / k])
            c = NB.case_surface(P, 2.0) @ R / k + shaft
            keep_out.append(("box", c.min(0) - 1.0 / k, c.max(0) + 1.0 / k))
            keep_out.append(("sphere", np.array([cx, shaft[1], shaft[2]]), rdome + (P.HIP_SEAM_GAP + 1.0) / k))
        if yaw_motor and not socket:            # hip yaw case in the belly over the leg (v14 layouts)
            import layout as LY
            ys, yc = LY.hip_yaw_case()
            cn = np.array([cx + yc[1] / k, cy - yc[0] / k, P.CUT_LEG_TOP + yc[2] / k])
            half = np.array([ys[1], ys[0], ys[2]]) / 2 / k + (P.WALL + 1.0) / k
            keep_out.append(("box", cn - half, cn + half))
    ref = np.mean([c for c in centres.values()], axis=0)            # leg centre line (x, y native)

    wall = int(np.ceil(P.WALL_MM / k / body.pitch))
    inner = [NB.Solid(ndimage.binary_erosion(s.grid, iterations=wall), s.lo, s.pitch) for s in (body, pack)]

    def inside(q):
        ok = inner[0].inside(q) | inner[1].inside(q)
        ok &= q[:, 2] >= zs + P.WALL_MM / k
        for kind, a, b in keep_out:
            if kind == "box":
                ok &= ~np.all((q >= a) & (q <= b), axis=1)
            else:
                ok &= np.linalg.norm(q - a, axis=1) > b
        return ok

    info = json.load(open(os.path.join(ROOT, "robot", v, "variant.json")))
    M = info["mass_kg"]
    g = np.linspace(-0.5, 0.5, 5)
    unit = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    xs = np.arange(-13.5, 13.6, 0.5)
    ys = np.arange(-16.0, 16.1, 0.5)
    zz = np.arange(zs, 45.0, 0.5)
    C = np.array(np.meshgrid(xs, ys, zz, indexing="ij")).reshape(3, -1).T
    placed = []
    for name, kg, size in parts:
        best = None
        for perm in {(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 0, 1), (1, 2, 0), (2, 1, 0)}:
            dims = np.array([size[i] for i in perm], float) / k          # native x (lateral), y (fore-aft), z
            # cost: height + forward offset of the part (native y: - is forward)
            cost = C[:, 2] - C[:, 1]
            order = np.argsort(cost)
            n = len(unit)
            for b0 in range(0, len(order), 2000):                       # cheapest spots first, in batches
                idx = order[b0:b0 + 2000]
                if best is not None and cost[idx[0]] >= best[0]:
                    break
                q = (C[idx][:, None, :] + unit[None] * dims).reshape(-1, 3)
                ok = inside(q).reshape(len(idx), n).all(axis=1)
                if ok.any():
                    i = idx[np.argmax(ok)]
                    if best is None or cost[i] < best[0]:
                        best = (cost[i], C[i].copy(), dims.copy())
                    break
        if best is None:                                   # report it and go on with the rest
            print(f"{name}: no room left for it")
            placed.append(dict(name=name, kg=kg, behind=None, up=None, side=None, size_mm=list(size), unplaced=True))
            continue
        _, c, dims = best
        keep_out.append(("box", c - dims / 2 - 1.0 / k, c + dims / 2 + 1.0 / k))
        placed.append(dict(name=name, kg=kg, behind=round(float(c[1] - ref[1]), 2), up=round(float(c[2] - P.CUT_LEG_TOP), 2),
                           side=round(float(c[0] - ref[0]), 2),
                           size_mm=[round(float(d * k), 1) for d in (dims[1], dims[0], dims[2])]))   # robot x, y, z
    placed.append(dict(name=WIRING[0], kg=WIRING[1], behind=0.0, up=12.0, side=0.0, size_mm=None))
    out = os.path.join(ROOT, "robot", v, "electronics_body_only.json" if "--body-only" in sys.argv
                       else "electronics.json")
    json.dump(placed, open(out, "w"), indent=2)
    for p in placed:
        if p.get("unplaced"):
            print(f"{p['name']:36s} {p['kg']*1000:4.0f} g  DOES NOT FIT ({p['size_mm']} mm)")
            continue
        print(f"{p['name']:36s} {p['kg']*1000:4.0f} g  {p['behind']*k:6.1f} mm behind, {p['up']*k:6.1f} up, "
              f"{p['side']*k:6.1f} left of the leg-centre line" + (f", {p['size_mm']} mm" if p['size_mm'] else ""))
    print("wrote", os.path.relpath(out, ROOT), f"(robot {M:.3f} kg)")


if __name__ == "__main__":
    main(sys.argv[1])          # add --body-only to keep everything out of the backpack
