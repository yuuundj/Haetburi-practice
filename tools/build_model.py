#!/usr/bin/env python3
"""Build one Haetburi variant: link meshes, URDF, MJCF scene, report.

Run:  HAETBURI_VARIANT=v6_split_round python3 tools/build_model.py
Out:  robot/<variant>/{haetburi.urdf, haetburi_scene.xml, meshes/, build_report.txt,
      variant.json}
All variants at once: python3 tools/build_all.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import haetburi_params as P
import massprops as MP
from segment import segment_exposed, segment_foot_both, segment_split
import layout as LY
import write_mjcf

MM = 1e-3
S = P.MESH_SCALE * MM           # native STL mm -> metres
DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
#  geometry helpers
# --------------------------------------------------------------------------- #
def to_link_frame(mesh, ty_native, link_origin, lift_mm):
    """Rotate the STL into the robot frame, scale to metres, move to a link frame.

    (x_w, y_w, z_w) = (-y_stl, x_stl - ty_native, z_stl) * S + (0, 0, lift)
    then subtract `link_origin` (metres, robot frame).
    """
    m = mesh.copy()
    v = m.vertices
    out = np.empty_like(v)
    out[:, 0] = -v[:, 1] * S
    out[:, 1] = (v[:, 0] - ty_native) * S
    out[:, 2] = v[:, 2] * S + lift_mm * MM
    m.vertices = out - np.asarray(link_origin, float)
    return m


def _fit_sphere(pts):
    c = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    return c, float(np.linalg.norm(pts - c, axis=1).max())


def _fit_box(pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return tuple(hi - lo), tuple((lo + hi) / 2.0)


def torso_collision(mesh, z_neck, y_arm, n_bands):
    """Chest box + two arm spheres + a stack of spheres up the head.

    A single band-sphere stack over the whole upper body is useless here: the
    arms make the chest bands ~300 mm wide, so the lowest sphere would scrape
    the floor while the robot is still standing upright.
    """
    v = mesh.vertices
    low = v[:, 2] < z_neck
    out = []
    chest = v[low & (np.abs(v[:, 1]) < y_arm)]
    out.append(("box", _fit_box(chest)))
    for sgn in (+1, -1):
        arm = v[low & (sgn * v[:, 1] >= y_arm)]
        if len(arm) > 8:
            c, r = _fit_sphere(arm)
            out.append(("sphere", (r, tuple(c))))
    head = v[~low]
    edges = np.linspace(head[:, 2].min(), head[:, 2].max(), n_bands + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = head[(head[:, 2] >= lo) & (head[:, 2] <= hi)]
        if len(sel) < 8:
            continue
        c, r = _fit_sphere(sel)
        out.append(("sphere", (r, tuple(c))))
    return out


# --------------------------------------------------------------------------- #
#  model assembly
# --------------------------------------------------------------------------- #
class Link:
    def __init__(self, name):
        self.name = name
        self.items = []          # (mass, com, I) triples, link frame
        self.visual = []         # ('box', (size, pos), material)
        self.collision = []      # ('box'|'sphere'|'mesh', params)
        self.mesh_file = None
        self.motors = []         # (size, centre) in metres, link frame
        self.z_sole_mm = None    # feet: sole height below the link origin

    def props(self):
        return MP.combine(self.items)


def _motor(link, size_mm, centre_mm):
    size, c = np.asarray(size_mm) * MM, np.asarray(centre_mm) * MM
    link.items.append(MP.box_props(P.MOTOR_MASS_KG, size, c))
    link.visual.append(("box", (tuple(size), tuple(c)), "motor"))
    link.motors.append((size, c))


def _leg_chain_joints(org, upper, lower):
    """Joint list; `upper`/`lower` name the leg->ankle and ankle->foot joints."""
    ax = {"ankle_pitch": (0, 1, 0), "ankle_roll": (1, 0, 0)}
    j = [("waist_yaw", "pelvis", "torso", (0, 0, 1), org["torso"] - org["pelvis"], "waist_yaw")]
    for sd in ("l", "r"):
        j.append((f"hip_pitch_{sd}", "pelvis", f"leg_{sd}", (0, 1, 0),
                  org[f"leg_{sd}"] - org["pelvis"], "hip_pitch"))
    for sd in ("l", "r"):
        j.append((f"{upper}_{sd}", f"leg_{sd}", f"ankle_{sd}", ax[upper],
                  org[f"ankle_{sd}"] - org[f"leg_{sd}"], upper))
    for sd in ("l", "r"):
        j.append((f"{lower}_{sd}", f"ankle_{sd}", f"foot_{sd}", ax[lower],
                  org[f"foot_{sd}"] - org[f"ankle_{sd}"], lower))
    return j


def _leg_centres(pieces):
    c_l = pieces["leg_l"].metadata["centre_native"]
    c_r = pieces["leg_r"].metadata["centre_native"]
    assert abs((c_l[0] - c_r[0]) / 2.0 - P.LEG_HALF_SPACING_NATIVE) < 0.3, \
        "update LEG_HALF_SPACING_NATIVE in haetburi_params.py"
    y_hip_mm = (c_l[0] - c_r[0]) / 2.0 * P.MESH_SCALE
    x_axis = -(c_l[1] + c_r[1]) / 2.0 * S
    return c_l, c_r, y_hip_mm, x_axis


def _torso(L, org, pieces, lift, waist_motor=True, battery_backpack=False):
    tr = to_link_frame(pieces["torso"], 0.0, org["torso"], lift)
    L["torso"].mesh_file = tr
    if "torso_parts" in pieces:                   # each part with its own printed wall (SHELL_WALL)
        for mesh, part in pieces["torso_parts"]:
            w = getattr(P, "SHELL_WALL", {}).get(part, P.WALL_MM)
            L["torso"].items.append(MP.shell_props(to_link_frame(mesh, 0.0, org["torso"], lift),
                                                   wall_m=w * P.RIB_FACTOR * 1e-3,
                                                   drop_planes=(0.0,) if part == "body" else ()))
    else:
        L["torso"].items.append(MP.shell_props(tr, drop_planes=(0.0,)))
    if waist_motor:
        ws, wc = LY.waist_case()
        _motor(L["torso"], ws, wc)
        L["torso"].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, 0)))
    L["torso"].items.append(MP.point_props(P.HEAD_EXTRA_KG, (0, 0, 0.62 * tr.bounds[1][2])))
    if battery_backpack:
        backpack = np.array([-33.0 * S, 0.0, 65.0 * S + lift * MM]) - org["torso"]
        L["torso"].items.append(MP.point_props(P.ELECTRONICS_KG, backpack))
    z_neck = (P.NECK_NATIVE * P.MESH_SCALE + lift) * MM - org["torso"][2]
    L["torso"].collision.extend(torso_collision(tr, z_neck, P.ARM_X_NATIVE * S, P.HEAD_BANDS))


def _pelvis_shell(L, org, pieces, lift, drop):
    pv = to_link_frame(pieces["pelvis"], 0.0, org["pelvis"], lift)
    L["pelvis"].mesh_file = pv
    L["pelvis"].items.append(MP.shell_props(pv, drop_planes=drop))
    belly = pv.vertices[np.abs(pv.vertices[:, 1]) < P.ARM_X_NATIVE * S]
    L["pelvis"].collision.append(("box", _fit_box(belly)))
    return pv


def _foot_bun(L, key, sgn, f, z_p, pitch_case, mirror_motor):
    fm = LY.foot_mesh(f)
    fm.vertices[:, 1] *= sgn
    if sgn < 0:
        fm.invert()                                   # mirroring flips the winding
    fm.apply_scale(MM)
    L[key].mesh_file = fm
    L[key].items.append(MP.shell_props(fm))
    L[key].items.append(MP.point_props(0.015, (f["cx"] * MM, sgn * f["cy"] * MM, -z_p * MM)))
    ps, pc = pitch_case
    _motor(L[key], ps, pc * np.array([1, sgn, 1]) if mirror_motor else pc)
    L[key].collision.append(("mesh", key))
    L[key].z_sole_mm = z_p


# ----------------------------------------------------- v1 / v2: exposed ankle
def build_exposed():
    """Ankle motors stacked under the original legs; the character is shrunk so
    the whole robot is TOTAL_HEIGHT.  Motor boxes are centred on their shafts."""
    pieces = segment_exposed()
    c_l, c_r, y_hip_mm, x_axis = _leg_centres(pieces)
    y_hip = y_hip_mm * MM
    lift = P.ANKLE_STACK
    z_hip = (P.ANKLE_STACK + P.CUT_LEG_TOP * P.MESH_SCALE) * MM
    z_waist = (P.ANKLE_STACK + P.CUT_WAIST * P.MESH_SCALE) * MM
    z_ap, z_ar = P.Z_ANKLE_PITCH * MM, P.Z_ANKLE_ROLL * MM
    org = {"pelvis": np.array([x_axis, 0.0, z_hip]), "torso": np.array([x_axis, 0.0, z_waist])}
    for sd, sg_ in (("l", 1), ("r", -1)):
        org[f"leg_{sd}"] = np.array([x_axis, sg_ * y_hip, z_hip])
        org[f"ankle_{sd}"] = np.array([x_axis, sg_ * y_hip, z_ap])
        org[f"foot_{sd}"] = np.array([x_axis, sg_ * y_hip, z_ar])
    L = {n: Link(n) for n in org}
    ml, mw, mh = P.MOTOR_H * MM, P.MOTOR_W * MM, P.MOTOR_D * MM   # 46.5, 28.5, 34

    def box(link, size, pos):
        link.items.append(MP.box_props(P.MOTOR_MASS_KG, size, pos))
        link.visual.append(("box", (tuple(size), tuple(pos)), "motor"))
        link.motors.append((np.asarray(size), np.asarray(pos, float)))

    _pelvis_shell(L, org, pieces, lift, drop=(0.0, z_waist - z_hip))
    for sg_ in (1, -1):
        box(L["pelvis"], (ml, mw, mh), (0, sg_ * y_hip, 0))
    zw = max(z_waist - z_hip - mw / 2.0, mh / 2.0 + 2 * MM + mw / 2.0)
    box(L["pelvis"], (ml, mh, mw), (0, 0, zw))
    L["pelvis"].items.append(MP.point_props(3 * P.BRACKET_MASS_KG + P.ELECTRONICS_KG,
                                            (0, 0, (z_waist - z_hip) * 0.4)))
    _torso(L, org, pieces, lift, waist_motor=False)

    for sd, sg_ in (("l", 1), ("r", -1)):
        key = f"leg_{sd}"
        c_native = (c_l if sg_ > 0 else c_r)[0]
        lg = to_link_frame(pieces[key], c_native - sg_ * y_hip_mm / P.MESH_SCALE, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg, drop_planes=(0.0,)))
        zp = z_ap - z_hip
        box(L[key], (ml, mw, mh), (0, 0, zp))
        L[key].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, zp + mh / 2)))
        lb = lg.bounds
        L[key].collision.append(("box", (tuple(lb[1] - lb[0]), tuple((lb[0] + lb[1]) / 2))))
        L[key].collision.append(("box", ((ml, mw, mh), (0, 0, zp))))

        akey = f"ankle_{sd}"
        zr = z_ar - z_ap
        rb = (mw, P.ROLL_Y * MM, P.ROLL_V * MM)
        box(L[akey], rb, (0, 0, zr))
        L[akey].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, zr / 2)))
        L[akey].collision.append(("box", (rb, (0, 0, zr))))

        fkey = f"foot_{sd}"
        fx, fy, ft = P.FOOT_PAD_X * MM, P.FOOT_PAD_Y * MM, P.FOOT_PAD_T * MM
        zc, yc = ft / 2.0 - z_ar, sg_ * P.FOOT_PAD_Y_OFFSET * MM
        L[fkey].items.append(MP.box_props(0.060, (fx, fy, ft), (0, yc, zc)))
        L[fkey].visual.append(("box", ((fx, fy, ft), (0, yc, zc)), "foot"))
        L[fkey].collision.append(("box", ((fx, fy, ft), (0, yc, zc))))
        L[fkey].z_sole_mm = P.Z_ANKLE_ROLL

    joints = _leg_chain_joints(org, "ankle_pitch", "ankle_roll")
    hz = dict(lift=lift, crotch=z_hip / MM, waist=z_waist / MM, hip=z_hip / MM,
              top=P.TOTAL_HEIGHT)
    meta = dict(org=org, heights=hz, exclude=[],
                summary={"ankle stack under the legs (mm)": P.ANKLE_STACK,
                         "roll motor": "upright" if P.ROLL_UPRIGHT else "flat",
                         "foot pad (mm)": f"{P.FOOT_PAD_X:.0f} x {P.FOOT_PAD_Y:.0f} x {P.FOOT_PAD_T:.0f}"})
    return L, joints, meta


# --------------------------------------------- v4: both ankle motors in the foot
def build_foot_both():
    pieces = segment_foot_both()
    c_l, c_r, y_hip_mm, x_axis = _leg_centres(pieces)
    an = LY.ankle_layout_both()
    f, z_p = an["foot"], an["z_pitch"]
    z_r = z_p + an["dz"]
    roll_size, roll_c = an["roll"]
    leg0 = to_link_frame(pieces["leg_l"], c_l[0] - y_hip_mm / P.MESH_SCALE,
                         (x_axis, y_hip_mm * MM, 0.0), -P.LEG_BOTTOM_NATIVE * P.MESH_SCALE)
    leg_rel = leg0.vertices / MM
    leg_rel = leg_rel[leg_rel[:, 2] < 25.0]
    z_min = roll_c[2] + roll_size[2] / 2 + P.LEG_ABOVE_ROLL_CASE
    z_lb = z_p + LY.leg_clear_height_fixed_roll(f, leg_rel, an["dz"], z_min)
    hz = LY.body_heights_from_leg_bottom(z_lb)
    lift = hz["lift"]
    org = {"pelvis": np.array([x_axis, 0.0, hz["hip"] * MM]),
           "torso": np.array([x_axis, 0.0, hz["waist"] * MM])}
    for sd, sg_ in (("l", 1), ("r", -1)):
        y = sg_ * y_hip_mm * MM
        org[f"leg_{sd}"] = np.array([x_axis, y, hz["hip"] * MM])
        org[f"ankle_{sd}"] = np.array([x_axis, y, z_r * MM])
        org[f"foot_{sd}"] = np.array([x_axis, y, z_p * MM])
    L = {n: Link(n) for n in org}

    _pelvis_shell(L, org, pieces, lift,
                  drop=((hz["crotch"] - hz["hip"]) * MM, (hz["waist"] - hz["hip"]) * MM))
    hs, hc = LY.hip_case()
    for sg_ in (1, -1):
        _motor(L["pelvis"], hs, hc + np.array([0, sg_ * y_hip_mm, 0]))
    L["pelvis"].items.append(MP.point_props(2 * P.BRACKET_MASS_KG, (0, 0, 0)))
    _torso(L, org, pieces, lift, battery_backpack=True)

    for sd, sg_ in (("l", 1), ("r", -1)):
        key = f"leg_{sd}"
        c_native = (c_l if sg_ > 0 else c_r)[0]
        lg = to_link_frame(pieces[key], c_native - sg_ * y_hip_mm / P.MESH_SCALE, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg, drop_planes=((hz["crotch"] - hz["hip"]) * MM,)))
        L[key].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, lg.bounds[0][2])))
        L[key].collision.append(("box", _fit_box(lg.vertices)))
        akey = f"ankle_{sd}"
        _motor(L[akey], roll_size, roll_c - np.array([0, 0, an["dz"]]))
        L[akey].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, -an["dz"] / 2 * MM)))
        _foot_bun(L, f"foot_{sd}", sg_, f, z_p, an["pitch"], mirror_motor=False)

    joints = _leg_chain_joints(org, "ankle_roll", "ankle_pitch")
    meta = dict(org=org, heights=hz, exclude=[("leg_l", "foot_l"), ("leg_r", "foot_r")],
                summary={"foot (mm)": f"{f['L']:.1f} x {f['W']:.1f} x {f['H']:.1f}",
                         "pitch / roll axis above sole (mm)": f"{z_p:.1f} / {z_r:.1f}",
                         "ankle gap (mm)": round(z_lb - f["H"], 1)})
    return L, joints, meta


# ---------------------------------------------- v5 / v6: roll in leg, pitch in foot
def build_split():
    fl = LY.foot_layout()
    f, z_p = fl["foot"], fl["z_pitch"]
    leg_half = LY.leg_section()
    leg_len, roll_above_bottom, _ = LY.leg_length()
    pieces = segment_split(leg_len, leg_half)
    c_l, c_r, y_hip_mm, x_axis = _leg_centres(pieces)

    bottom_native = pieces["leg_l"].metadata["bottom_native"]
    leg0 = to_link_frame(pieces["leg_l"], c_l[0] - y_hip_mm / P.MESH_SCALE,
                         (x_axis, y_hip_mm * MM, 0.0), -bottom_native * P.MESH_SCALE)
    leg_rel = leg0.vertices / MM
    leg_rel = leg_rel[leg_rel[:, 2] < 30.0]
    off = float(P.PITCH_AXIS_X) if P.PITCH_AXIS_X != "auto" else 0.0   # main() resolves "auto"
    leg_rel = leg_rel - np.array([off, 0.0, 0.0])      # the leg seen from the pitch axis
    z_lb = z_p + LY.leg_clear_height(f, leg_rel, roll_above_bottom)
    z_r = z_lb + roll_above_bottom
    hz = LY.body_heights(z_lb, leg_len)
    lift = hz["lift"]
    org = {"pelvis": np.array([x_axis, 0.0, hz["hip"] * MM]),
           "torso": np.array([x_axis, 0.0, hz["waist"] * MM])}
    for sd, sg_ in (("l", 1), ("r", -1)):
        y = sg_ * y_hip_mm * MM
        org[f"leg_{sd}"] = np.array([x_axis, y, hz["hip"] * MM])
        org[f"ankle_{sd}"] = np.array([x_axis, y, z_r * MM])
        org[f"foot_{sd}"] = np.array([x_axis + off * MM, y, z_p * MM])
    L = {n: Link(n) for n in org}

    shorts = P.TRUNK_BLEND_TOP is not None
    drop = ((hz["waist"] - hz["hip"]) * MM,) if shorts else \
        ((hz["crotch"] - hz["hip"]) * MM, (hz["waist"] - hz["hip"]) * MM)
    _pelvis_shell(L, org, pieces, lift, drop=drop)
    hs, hc = LY.hip_case()
    for sg_ in (1, -1):
        _motor(L["pelvis"], hs, hc + np.array([0, sg_ * y_hip_mm, 0]))
    L["pelvis"].items.append(MP.point_props(2 * P.BRACKET_MASS_KG, (0, 0, 0)))
    if P.ELECTRONICS_AT == "belly":
        L["pelvis"].items.append(MP.point_props(P.ELECTRONICS_KG, np.array(P.ELECTRONICS_POS) * MM))
    _torso(L, org, pieces, lift, battery_backpack=(P.ELECTRONICS_AT == "backpack"))

    ls, lc = LY.leg_motor()
    span = (z_r - z_p) * MM
    for sd, sg_ in (("l", 1), ("r", -1)):
        key = f"leg_{sd}"
        c_native = (c_l if sg_ > 0 else c_r)[0]
        lg = to_link_frame(pieces[key], c_native - sg_ * y_hip_mm / P.MESH_SCALE, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg, drop_planes=((hz["crotch"] - hz["hip"]) * MM,)))
        _motor(L[key], ls, lc + np.array([0, 0, z_r - hz["hip"]]))
        L[key].collision.append(("box", _fit_box(lg.vertices)))
        # ankle bracket between the roll horn (leg) and the pitch horn (foot)
        L[f"ankle_{sd}"].items.append(
            MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.045, span), (off * MM / 2, 0, -span / 2)))
        _foot_bun(L, f"foot_{sd}", sg_, f, z_p, fl["pitch"], mirror_motor=True)

    joints = _leg_chain_joints(org, "ankle_roll", "ankle_pitch")
    meta = dict(org=org, heights=hz, exclude=[("leg_l", "foot_l"), ("leg_r", "foot_r")],
                summary={"foot (mm)": f"{f['L']:.1f} x {f['W']:.1f} x {f['H']:.1f}",
                         "leg section (mm)": f"{2*leg_half[1]:.1f} wide x {2*leg_half[0]:.1f} deep",
                         "leg length, crotch to bottom (mm)": round(leg_len, 1),
                         "pitch / roll axis above sole (mm)": f"{z_p:.1f} / {z_r:.1f}",
                         "ankle gap (mm)": round(z_lb - f["H"], 1)})
    return L, joints, meta


# ------------------------------------------ v7: no ankle pitch, roll motor in the foot
def build_roll_foot():
    pieces = segment_foot_both()                       # original leg, straight at the bottom
    c_l, c_r, y_hip_mm, x_axis = _leg_centres(pieces)
    fl = LY.foot_layout_roll()
    f, z_r = fl["foot"], fl["z_roll"]                  # roll axis height above the sole
    rs, rc = fl["roll"]
    leg0 = to_link_frame(pieces["leg_l"], c_l[0] - y_hip_mm / P.MESH_SCALE,
                         (x_axis, y_hip_mm * MM, 0.0), -P.LEG_BOTTOM_NATIVE * P.MESH_SCALE)
    leg_rel = leg0.vertices / MM
    leg_rel = leg_rel[leg_rel[:, 2] < 25.0]
    z_min = rc[2] + rs[2] / 2 + P.LEG_ABOVE_ROLL_CASE
    z_lb = z_r + LY.leg_clear_height_roll(f, leg_rel, z_min)
    hz = LY.body_heights_from_leg_bottom(z_lb)
    lift = hz["lift"]
    yaw = P.HIP_JOINT == "yaw"
    if yaw:                                            # yaw axis is vertical: put the joint at the crotch
        hz["hip"] = hz["crotch"]
    org = {"pelvis": np.array([x_axis, 0.0, hz["hip"] * MM]),
           "torso": np.array([x_axis, 0.0, hz["waist"] * MM])}
    for sd, sg_ in (("l", 1), ("r", -1)):
        y = sg_ * y_hip_mm * MM
        org[f"leg_{sd}"] = np.array([x_axis, y, hz["hip"] * MM])
        org[f"foot_{sd}"] = np.array([x_axis, y, z_r * MM])
    L = {n: Link(n) for n in org}

    _pelvis_shell(L, org, pieces, lift,
                  drop=((hz["crotch"] - hz["hip"]) * MM, (hz["waist"] - hz["hip"]) * MM))
    hs, hc = LY.hip_yaw_case() if yaw else LY.hip_case()
    for sg_ in (1, -1):
        _motor(L["pelvis"], hs, hc + np.array([0, sg_ * y_hip_mm, 0]))
    L["pelvis"].items.append(MP.point_props(2 * P.BRACKET_MASS_KG, (0, 0, 0)))
    if yaw:                                            # shaft extensions from the raised hip cases
        for sg_ in (1, -1):
            L["pelvis"].items.append(MP.point_props(P.BRACKET_MASS_KG,
                                                    (0, sg_ * y_hip_mm * MM, P.HIP_YAW_LIFT / 2 * MM)))
    if P.ELECTRONICS_AT == "belly":
        L["pelvis"].items.append(MP.point_props(P.ELECTRONICS_KG, np.array(P.ELECTRONICS_POS) * MM))
    _torso(L, org, pieces, lift, battery_backpack=(P.ELECTRONICS_AT == "backpack"))
    if P.ELECTRONICS_AT == "chest":
        L["torso"].items.append(MP.point_props(P.ELECTRONICS_KG, np.array(P.ELECTRONICS_POS) * MM))

    for sd, sg_ in (("l", 1), ("r", -1)):
        key = f"leg_{sd}"
        c_native = (c_l if sg_ > 0 else c_r)[0]
        lg = to_link_frame(pieces[key], c_native - sg_ * y_hip_mm / P.MESH_SCALE, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg, drop_planes=((hz["crotch"] - hz["hip"]) * MM,)))
        # bracket from the leg bottom down to the roll horn in the foot
        span = (z_lb - z_r) * MM
        L[key].items.append(MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.040, span),
                                         (0, 0, lg.bounds[0][2] - span / 2)))
        L[key].collision.append(("box", _fit_box(lg.vertices)))
        _foot_bun(L, f"foot_{sd}", sg_, f, z_r, (rs, rc), mirror_motor=True)

    joints = [("waist_yaw", "pelvis", "torso", (0, 0, 1), org["torso"] - org["pelvis"], "waist_yaw")]
    hip, hax = ("hip_yaw", (0, 0, 1)) if yaw else ("hip_pitch", (0, 1, 0))
    for sd in ("l", "r"):
        joints.append((f"{hip}_{sd}", "pelvis", f"leg_{sd}", hax,
                       org[f"leg_{sd}"] - org["pelvis"], hip))
    for sd in ("l", "r"):
        joints.append((f"ankle_roll_{sd}", f"leg_{sd}", f"foot_{sd}", (1, 0, 0),
                       org[f"foot_{sd}"] - org[f"leg_{sd}"], "ankle_roll"))
    meta = dict(org=org, heights=hz, exclude=[("leg_l", "foot_l"), ("leg_r", "foot_r")],
                summary={"actuated joints": f"5 (waist yaw, {hip.replace('_', ' ')} x2, ankle roll x2)",
                         "foot (mm)": f"{f['L']:.1f} x {f['W']:.1f} x {f['H']:.1f}",
                         "roll axis above sole (mm)": round(z_r, 1),
                         "leg shell (mm)": f"original, {(P.CUT_LEG_TOP - P.LEG_BOTTOM_NATIVE) * P.MESH_SCALE:.1f} from bottom to crotch",
                         "ankle gap (mm)": round(z_lb - f["H"], 1),
                         "ankle pitch axis ahead of the leg (mm)": round(off, 1)})
    return L, joints, meta

# ------------------- v11: v9's joints on the new display character, roll motor in its own feet
def build_newbase():
    """The new character (make_display_model: tail on, fists on the display) cut at
    the crotch and the waist.  Its STL feet keep their shape; MESH_SCALE is chosen so
    a walled XL430 lies inside each (newbase.fit_motor), shaft fore-aft = roll axis.
    The legs and everything above rise until the legs clear the feet over the
    roll range.  All motors are checked against voxel solids of the open STL."""
    import trimesh
    import make_display_model as MD
    import newbase as NB
    k = P.MESH_SCALE
    parts, _, dc, _ = MD.build(P.CHAR_HEIGHT, log=lambda *a: None,
                               display_mm=(P.DISPLAY_OUTLINE[1], P.DISPLAY_OUTLINE[2], P.DISPLAY_OUTLINE[0]))
    for n, faces in (("head", 50000), ("backpack", 35000)):       # MuJoCo takes <= 200k faces a mesh
        a0 = parts[n].area
        parts[n] = parts[n].simplify_quadric_decimation(face_count=faces)
        assert abs(parts[n].area / a0 - 1) < 0.02, f"decimating {n} changed its area"
    surf = NB.part_surfaces()
    for n in ("foot_a", "foot_b"):                                 # flat soles (drawn without the
        parts[n] = NB.flat_sole(parts[n], P.FOOT_FLAT_Z, P.FOOT_FILLET)   # loose patches, which flicker)
        surf[n] = NB.flat_sole(surf[n], P.FOOT_FLAT_Z, P.FOOT_FILLET)
    sol = {n: NB.solid(surf[n]) for n in ("foot_a", "foot_b", "body_legs", "backpack")}

    bl = surf["body_legs"]
    pts, _ = trimesh.sample.sample_surface(bl, 400000, seed=2)
    pts = np.vstack([pts, bl.vertices])
    style = getattr(P, "FOOT_STYLE", "inside")
    slipper, two, stacked = style == "slipper", style == "inside2", style == "stacked"
    feet, legs, lifts, extra = {}, {}, [], {}
    for sd, name, sgn in (("l", "foot_a", 1), ("r", "foot_b", -1)):
        lp = pts[(pts[:, 2] < P.CUT_LEG_TOP) & (sgn * pts[:, 0] > 0)]
        band = lp[lp[:, 2] > P.CUT_LEG_TOP - 1.5]
        if slipper:
            lp = lp[lp[:, 2] >= P.LEG_CUT_Z]                       # the leg ends in a flat cap:
            import shapely.geometry as sg                          # its face counts for clearance too
            rim = sg.MultiPoint(lp[lp[:, 2] < P.LEG_CUT_Z + 0.3][:, :2]).convex_hull
            gx, gy = np.meshgrid(np.arange(rim.bounds[0], rim.bounds[2], 0.1),
                                 np.arange(rim.bounds[1], rim.bounds[3], 0.1))
            face = np.c_[gx.ravel(), gy.ravel()]
            face = face[[rim.contains(sg.Point(q)) for q in face]]
            lp = np.vstack([lp, np.c_[face, np.full(len(face), P.LEG_CUT_Z)]])
        legs[sd] = ((band[:, 0].min() + band[:, 0].max()) / 2, (band[:, 1].min() + band[:, 1].max()) / 2,
                    lp[:, 2].min())
        if stacked:
            # one ankle motor in the STL foot, the other stacked above it in the leg,
            # the leg lengthened by its own section (STACK_UP names the upper joint)
            if P.STACK_UP == "roll":       # pitch motor lying in the foot, long side to the toe,
                RL = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # axis under the CoM
                cand = NB._placements(P, sol[name], sol[name].grid, RL, k)
                if not len(cand):
                    raise RuntimeError(f"the ankle pitch motor does not fit {name} at x{k}")
                ty = legs[sd][1] - P.PITCH_AXIS_AHEAD_NATIVE
                ref = feet["l"][1] * [-1, 1, 1] if sd == "r" else np.array([cand[:, 0].mean(), 0, 0])
                sL = cand[np.argmin(np.abs(cand[:, 1] - ty) + 0.02 * np.abs(cand[:, 0] - ref[0])
                                    + 0.02 * cand[:, 2])]
            else:                          # roll motor lying in the foot, long end outward (v11)
                th = sgn * P.FOOT_MOTOR_THETA
                RL = np.stack(NB.motor_frame(th))
                sL = NB.fit_motor(P, sol[name], k, th)
                if sL is None:
                    raise RuntimeError(f"the ankle roll motor does not fit {name} at x{k}")
            feet[sd] = (name, sL, RL)
            continue
        if two:
            # ankle roll motor standing in the foot (A); ankle pitch motor (B) on the
            # roll output, lying fore-aft toward the toe; the leg rides B's horn
            RA = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])           # shaft fore-aft, long side up
            hB = 1.0 if P.PITCH_LONG_END == "heel" else -1.0
            RB = np.array([[1.0, 0, 0], [0, hB, 0], [0, 0, hB]])             # shaft lateral, long side fore-aft
            r_roll, r_pitch = P.LIM["ankle_roll"][1], P.LIM["ankle_pitch"][1]
            # the ankle pitch axis goes under the robot's centre of mass (v10's lesson:
            # otherwise the stance ankles hold the robot up all the time)
            fit = None
            if sd == "r":                  # the feet are near mirror images: mirror the left one
                mA, mB = feet["l"][1] * [-1, 1, 1], feet["l"][2] * [-1, 1, 1]
                if NB.two_fit_ok(P, sol[name], k, RA, RB, mA, mB, (0, 1, 0), r_roll):
                    fit = (mA, mB)
            if fit is None:
                fit = NB.fit_two(P, sol[name], k, RA, RB, (0, 1, 0), r_roll, n_try=60,
                                 b_target_y=legs[sd][1] - P.PITCH_AXIS_AHEAD_NATIVE,
                                 a_hint=None if sd == "l" else mA)
            if fit is None:
                raise RuntimeError(f"the two ankle motors do not fit {name} at x{k}")
            sA, sB = fit
            fb = sol[name]
            foot_obst = NB.union_solid((fb.lo, fb.lo + np.array(fb.grid.shape) * fb.pitch),
                                       [fb.inside, NB.in_case(P, RA, sA, k)])
            bl_, bh_ = NB.case_box_robot(P, RB, k)
            half = np.array([bl_[1], bl_[0], bl_[2]]) / 2 / k + 0.2
            cb = sB + np.array([bh_[1], -bh_[0], bh_[2]]) / k
            ankle_obst = NB.union_solid((cb - half, cb + half), [NB.in_case(P, RB, sB, k)])
            feet[sd] = (name, sA, sB, RA, RB)
            lifts.append(NB.leg_lift_two(P, lp, foot_obst, ankle_obst, sA, (0, 1, 0), r_roll,
                                         sB, (1, 0, 0), r_pitch, k))
            continue
        if not slipper:
            # roll motor inside the foot, fitted on its own, long end outward
            th = sgn * P.FOOT_MOTOR_THETA
            shaft = NB.fit_motor(P, sol[name], k, th)
            if shaft is None:
                raise RuntimeError(f"the roll motor does not fit {name} at x{k}")
            feet[sd] = (name, th, shaft)
            lifts.append(NB.leg_lift(P, lp, sol[name], shaft, k))
            continue
        # slipper: the roll motor lies on a sole plate at the heel, out in the open,
        # long end outward, horn toward the heel; only the front of the foot is kept,
        # as a toe cap fixed to the plate
        foot = parts[name]
        fb = foot.bounds
        cx = (fb[0][0] + fb[1][0]) / 2.0
        standing = P.ANKLE_MOTOR == "standing"
        y_back = fb[1][1] - P.HEEL_INSET / k - P.MOTOR_HORN / k           # case back (horn behind it)
        if standing:                       # long side up, shaft near the bottom, centred on the foot
            inset = (P.ANKLE_INSET if P.ANKLE_INSET != "max"
                     else (fb[1][0] - fb[0][0]) / 2 * k - P.MOTOR_W / 2 - 1.0)
            cx_m = cx - sgn * inset / k                                   # slid toward the inner edge
            shaft = np.array([cx_m, y_back - P.MOTOR_D / 2 / k,
                              (P.SOLE_PLATE + P.MOTOR_AXIS_FROM_END) / k])
            xs = [cx_m - P.MOTOR_W / 2 / k, cx_m + P.MOTOR_W / 2 / k]
            ztop = (P.SOLE_PLATE + P.MOTOR_H) / k
        else:                              # lying, long end outward
            shaft = np.array([cx - sgn * LY.H_MID / k, y_back - P.MOTOR_D / 2 / k,
                              (P.SOLE_PLATE + P.MOTOR_W / 2) / k])
            xs = sorted([shaft[0] - sgn * P.MOTOR_AXIS_FROM_END / k,
                         shaft[0] + sgn * (P.MOTOR_H - P.MOTOR_AXIS_FROM_END) / k])
            ztop = (P.SOLE_PLATE + P.MOTOR_W) / k
        box_lo = np.array([xs[0], shaft[1] - P.MOTOR_D / 2 / k, P.SOLE_PLATE / k])
        box_hi = np.array([xs[1], shaft[1] + (P.MOTOR_D / 2 + P.MOTOR_HORN) / k, ztop])
        cap_ = foot.slice_plane([0, P.SLIPPER_CUT_Y, 0], [0, -1, 0])
        sole = NB.outline(foot, 1e-3)
        import shapely.geometry as sg
        plate_poly = sole.intersection(sg.box(-1e3, P.SLIPPER_CUT_Y - 0.5, 1e3, 1e3))
        plate = trimesh.creation.extrude_polygon(plate_poly, P.SOLE_PLATE / k)
        cap_solid = NB.solid(cap_)
        obst = NB.union_solid((np.minimum(fb[0], box_lo), np.maximum(fb[1], box_hi)),
                              [cap_solid.inside, NB.in_box(box_lo, box_hi),
                               NB.in_box([*plate.bounds[0][:2], 0], plate.bounds[1])])
        feet[sd] = (name, sgn * 90.0, shaft)
        extra[sd] = dict(cap=cap_, plate=plate, box=(box_lo, box_hi), cap_solid=cap_solid)
        # the leg rides the horn; the bracket plate under the leg needs its thickness too
        lifts.append(NB.leg_lift(P, lp, obst, shaft, k, clear_mm=P.LEG_FOOT_CLEAR + P.BRACKET_T))
    if stacked:
        import shapely.geometry as sg
        from shapely import contains_xy
        up_roll = P.STACK_UP == "roll"
        # upper motor: roll standing (shaft fore-aft, long side up), or pitch standing
        # (shaft lateral, long side up: lying, its walled footprint is too big for the
        # leg's rounded section and the corners poke out)
        RU = (np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]]) if up_roll
              else np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]))
        tall = P.MOTOR_H
        leg_len = (P.WALL + tall + P.LEG_TOP_CLEAR) / k             # leg bottom to crotch, native
        z_b = P.CUT_LEG_TOP - leg_len
        from shapely import affinity
        for sd, sgn in (("l", 1), ("r", -1)):
            near = pts[(np.abs(pts[:, 2] - P.LEG_CUT_Z) < 0.15) & (sgn * pts[:, 0] > 0)]
            poly0 = sg.MultiPoint(near[:, :2]).convex_hull.buffer(-P.LEG_JOIN_INSET)   # the leg's own section
            cx, cy, _ = legs[sd]
            ext, poly = 0.0, poly0
            if up_roll:
                up_pt = np.array([cx, cy, z_b + (P.WALL + P.MOTOR_AXIS_FROM_END) / k])
            else:                          # pitch axis as near under the CoM as the leg allows,
                # the case's footprint (with walls, horn outward) wholly inside the leg outline;
                # LEG_FRONT_EXT_MAX > 0 lets the added part of the leg grow forward (only as
                # much as needed) so the axis can reach the CoM
                hx_in = (P.MOTOR_D / 2 + P.WALL) / k
                hx_out = (P.MOTOR_D / 2 + P.MOTOR_HORN + P.WALL) / k
                fr = bk = (P.MOTOR_W / 2 + P.WALL) / k
                ty = cy - P.PITCH_AXIS_AHEAD_NATIVE
                while True:
                    best = None
                    for x in np.arange(cx - 1.5, cx + 1.5, 0.05):
                        for y in np.arange(poly.bounds[1], poly.bounds[3], 0.05):
                            xs_ = sorted([x - sgn * hx_in, x + sgn * hx_out])
                            if poly.contains(sg.box(xs_[0], y - fr, xs_[1], y + bk)):
                                cost = abs(y - ty) + 0.1 * abs(x - cx)
                                if best is None or cost < best[0]:
                                    best = (cost, x, y)
                    if (best is not None and abs(best[2] - ty) <= 0.05) or ext >= P.LEG_FRONT_EXT_MAX:
                        break
                    ext += 0.25
                    poly = sg.MultiPoint(list(poly0.exterior.coords)
                                         + list(affinity.translate(poly0, 0, -ext).exterior.coords)).convex_hull
                if best is None:
                    raise RuntimeError("the ankle pitch motor does not fit the leg section")
                up_pt = np.array([best[1], best[2], z_b + (P.WALL + P.MOTOR_AXIS_FROM_END) / k])
            # it runs LEG_JOIN_OVERLAP up inside the original leg: no slit at the join; when it
            # reaches forward, all the way to the crotch, so it also covers the motor's top
            z_top = P.CUT_LEG_TOP if ext > 0 else P.LEG_CUT_Z + P.LEG_JOIN_OVERLAP
            prism = trimesh.creation.extrude_polygon(poly, z_top - z_b)
            prism.apply_translation([0, 0, z_b])
            side, _ = trimesh.sample.sample_surface(prism, 20000, seed=3)
            name, sL, RL = feet[sd]
            fsol = sol[name]
            obst = NB.union_solid((fsol.lo, fsol.lo + np.array(fsol.grid.shape) * fsol.pitch),
                                  [fsol.inside, NB.in_case(P, RL, sL, k)])
            lifts.append(NB.leg_lift_stacked(P, side, up_pt, obst, sL, k, up=P.STACK_UP))
            legs[sd] = (cx, cy, z_b)
            extra[sd] = dict(prism=prism, up_pt=up_pt, RU=RU, poly=poly, front_ext=ext)
    dz = max(lifts)
    lift = dz * k
    hip_pitch = getattr(P, "HIP_PITCH", False)
    hip_yaw = getattr(P, "HIP_YAW", True)          # v20: hip pitch without hip yaw
    seam_kind = getattr(P, "HIP_SEAM", "plane") if hip_pitch else "plane"
    sphere = seam_kind in ("sphere", "carve", "socket")     # hip yaw and pitch axes meet
    hp, seam = {}, None
    if hip_pitch:
        # hip pitch motor in the belly under each hip yaw motor (the legs stay the STL's)
        for sd, sgn in (("l", 1), ("r", -1)):
            fit = NB.fit_hip_pitch(P, sol["body_legs"], k, legs[sd], P.CUT_LEG_TOP, sgn,
                                   float(P.HIP_PITCH_AHEAD), P.LIM["hip_yaw"][1] if hip_yaw else 0.0,
                                   yaw_at_shaft=sphere)
            if fit is None:
                raise RuntimeError(f"the hip pitch motor does not fit the belly at x{k}")
            hp[sd] = fit
        if seam_kind == "sphere":
            # leg / belly seam on a sphere about the point where the hip axes meet
            cen = {sd: np.array([legs[sd][0], hp[sd][0][1], hp[sd][0][2]]) for sd in hp}
            seam = NB.hip_seam(P, parts["body_legs"], sol["body_legs"], cen, k, P.HIP_SEAM_GAP,
                               P.HIP_PITCH_TARGET + 2.0)
            seam["centres"] = cen
        # the hip yaw motor above it: horn, then the bracket onto the pitch case
        lo_, hi_ = NB.case_extent(P, walls=False)
        P.HIP_YAW_LIFT = max((hp[sd][0][2] - P.CUT_LEG_TOP) * k + hi_[1] for sd in hp) \
            + P.MOTOR_HORN + P.BRACKET_T + 1.0
        if seam_kind == "carve":
            # v18: legs cut off whole at the crotch, the lower belly carved around their sweep
            cen = {sd: np.array([legs[sd][0], hp[sd][0][1], hp[sd][0][2]]) for sd in hp}
            boxes = {}
            ys, yc = LY.hip_yaw_case()
            for sd in hp:
                c_ = NB.case_surface(P, 2.0) @ hp[sd][1] / k + hp[sd][0]      # walled pitch case
                yc_n = np.array([cen[sd][0] + yc[1] / k, cen[sd][1] - yc[0] / k, P.CUT_LEG_TOP + yc[2] / k])
                ys_n = np.array([ys[1], ys[0], ys[2]]) / k
                boxes[sd] = [(c_.min(0), c_.max(0), "pitch"), (yc_n - ys_n / 2, yc_n + ys_n / 2, "yaw")]
            seam = NB.hip_carve(P, parts["body_legs"], sol["body_legs"], cen, k, P.HIP_SEAM_GAP,
                                P.HIP_CARVE_PITCH, P.HIP_CARVE_YAW, boxes)
            seam["centres"], seam["R"] = cen, None
        if seam_kind == "socket":
            # v19 (PartsAssemblyV2): flat torso bottom just under the hip pitch cases, legs
            # carried up into sockets and closed by a dome about the hip pivot
            cen = {sd: np.array([legs[sd][0], hp[sd][0][1], hp[sd][0][2]]) for sd in hp}
            if getattr(P, "HIP_SOCKET_AT", "pivot") == "pivot":   # the bottom through the hip pivot:
                zs = min(hp[sd][0][2] for sd in hp)                   # the leg rim barely leaves it
            else:                                                     # just under the hip pitch cases
                lo_w, _ = NB.case_extent(P)
                zs = min(hp[sd][0][2] for sd in hp) + lo_w[1] / k - 0.5 / k
            ys, yc = LY.hip_yaw_case()
            ybox = {}
            for sd in hp:
                yc_n = np.array([cen[sd][0] + yc[1] / k, cen[sd][1] - yc[0] / k, P.CUT_LEG_TOP + yc[2] / k])
                ys_n = np.array([ys[1], ys[0], ys[2]]) / k
                ybox[sd] = (yc_n - ys_n / 2, yc_n + ys_n / 2)
            uni = stacked and getattr(P, "LEG_UNIFORM", False)
            if getattr(P, "TORSO_FIT", False):
                # v21: the lower torso re-lofted to cover the legs (they no longer stand
                # wider than a narrowing belly)
                polys = {sd: extra[sd]["poly"] for sd in hp} if stacked else None
                new, _ = NB.fit_torso_bottom(parts["body_legs"], zs, zs + P.TORSO_FIT_BLEND, polys,
                                             (P.WALL_MM + 1.5) / k)
                parts["body_legs"], surf["body_legs"] = new, new
                sol["body_legs"] = NB.solid(new)
                new.export(os.path.join(P.OUT_DIR, "body_native.stl"))           # for place_electronics
            seam = NB.hip_socket(P, parts["body_legs"], sol["body_legs"], cen, k, P.HIP_SEAM_GAP,
                                 P.HIP_CARVE_PITCH, P.HIP_CARVE_YAW, zs,
                                 {sd: (hp[sd][1], hp[sd][0]) for sd in hp}, ybox if hip_yaw else None,
                                 leg_polys={sd: extra[sd]["poly"] for sd in hp} if uni else None,
                                 leg_bottom={sd: legs[sd][2] for sd in hp} if uni else None)
            seam["uniform"] = uni
            seam["centres"], seam["R"], seam["z_split"] = cen, None, zs
    top = max(p.bounds[1][2] for p in parts.values())
    hz = dict(lift=lift, crotch=P.CUT_LEG_TOP * k + lift, waist=P.CUT_WAIST * k + lift,
              top=top * k + lift)
    hz["hip"] = hz["crotch"]
    x_axis = -(legs["l"][1] + legs["r"][1]) / 2.0 * S
    org = {"pelvis": np.array([x_axis, 0.0, hz["crotch"] * MM]),
           "torso": np.array([x_axis, 0.0, hz["waist"] * MM])}
    legc = {}
    for sd in ("l", "r"):
        cx, cy, _ = legs[sd]
        org[f"leg_{sd}"] = np.array([-cy * S, cx * S, hz["crotch"] * MM])
        legc[sd] = org[f"leg_{sd}"].copy()            # leg centre at the crotch
        if hip_pitch:                  # hip link on the yaw shaft; the leg hangs from the pitch shaft
            sh = hp[sd][0]
            if hip_yaw:
                org[f"hip_{sd}"] = org[f"leg_{sd}"].copy()
                if sphere:             # the yaw axis goes through the pitch shaft
                    org[f"hip_{sd}"][0] = -sh[1] * S
            org[f"leg_{sd}"] = np.array([-sh[1] * S, sh[0] * S, sh[2] * S + lift * MM])
        if two:
            _, sA, sB, _, _ = feet[sd]
            org[f"ankle_{sd}"] = np.array([-sB[1] * S, sB[0] * S, sB[2] * S])
            org[f"foot_{sd}"] = np.array([-sA[1] * S, sA[0] * S, sA[2] * S])
        elif stacked:
            up_ = extra[sd]["up_pt"]
            sL = feet[sd][1]
            org[f"ankle_{sd}"] = np.array([-up_[1] * S, up_[0] * S, up_[2] * S + lift * MM])
            org[f"foot_{sd}"] = np.array([-sL[1] * S, sL[0] * S, sL[2] * S])
        else:
            sx, sy, sz = feet[sd][2]
            org[f"foot_{sd}"] = np.array([-sy * S, sx * S, sz * S])
    L = {n: Link(n) for n in org}

    body = parts["body_legs"]
    below = body.slice_plane([0, 0, P.CUT_LEG_TOP], [0, 0, -1])
    pieces = {
        "torso": trimesh.util.concatenate(
            [body.slice_plane([0, 0, P.CUT_WAIST], [0, 0, 1])]
            + [parts[n] for n in ("head", "backpack", "arm_a", "arm_b", "tail")]),
        "pelvis": body.slice_plane([0, 0, P.CUT_LEG_TOP], [0, 0, 1]).slice_plane([0, 0, P.CUT_WAIST], [0, 0, -1]),
        "leg_l": below.slice_plane([0, 0, 0], [1, 0, 0]),
        "leg_r": below.slice_plane([0, 0, 0], [-1, 0, 0]),
    }
    top_body = body.slice_plane([0, 0, P.CUT_WAIST], [0, 0, 1])
    if seam is not None:                          # legs cut off along their spheres
        up = seam["upper"]
        top_body = up.slice_plane([0, 0, P.CUT_WAIST], [0, 0, 1])
        pieces["torso"] = trimesh.util.concatenate(
            [top_body] + [parts[n] for n in ("head", "backpack", "arm_a", "arm_b", "tail")])
        pieces["pelvis"] = up.slice_plane([0, 0, P.CUT_WAIST], [0, 0, -1])
        pieces["leg_l"], pieces["leg_r"] = seam["legs"]["l"], seam["legs"]["r"]
    pieces["torso_parts"] = [(top_body, "body")] + [(parts[n], n.split("_")[0])
                                                    for n in ("head", "backpack", "arm_a", "arm_b", "tail")]
    if stacked and not (seam is not None and seam.get("uniform")):   # original leg above its open end + its own section, longer
        for key, sd in (("leg_l", "l"), ("leg_r", "r")):
            lg = pieces[key].slice_plane([0, 0, P.LEG_CUT_Z], [0, 0, 1])
            pieces[key] = trimesh.util.concatenate([lg, extra[sd]["prism"]])
    if slipper:                                   # the leg's open bottom gets a flat cap
        for key in ("leg_l", "leg_r"):
            lg = pieces[key].slice_plane([0, 0, P.LEG_CUT_Z], [0, 0, 1])
            pieces[key] = trimesh.util.concatenate([lg, NB.cap(NB.outline(lg, P.LEG_CUT_Z + 1e-3),
                                                                  P.LEG_CUT_Z, up=False)])
    _pelvis_shell(L, org, pieces, lift, drop=())
    hs, hc = LY.hip_yaw_case()
    for sd in ("l", "r"):
        if hip_yaw:
            yaw_at = org[f"hip_{sd}" if hip_pitch else f"leg_{sd}"]
            _motor(L["pelvis"], hs, hc + (yaw_at - org["pelvis"]) / MM)
        if hip_pitch and hip_yaw:
            d_ = org[f"leg_{sd}"] - org[f"hip_{sd}"]
            ps, pc = NB.case_box_robot(P, hp[sd][1], k)
            _motor(L[f"hip_{sd}"], ps, pc + d_ / MM)
            L[f"hip_{sd}"].items.append(MP.point_props(P.BRACKET_MASS_KG, d_ / 2 + [0, 0, 0.01]))   # yaw horn to pitch case
        elif hip_pitch:                # v20: the hip pitch motor fixed in the belly
            ps, pc = NB.case_box_robot(P, hp[sd][1], k)
            _motor(L["pelvis"], ps, pc + (org[f"leg_{sd}"] - org["pelvis"]) / MM)
    L["pelvis"].items.append(MP.point_props(2 * P.BRACKET_MASS_KG, (0, 0, 0)))
    if getattr(P, "ELECTRONICS_ITEMS", None):                    # parts placed where they fit
        for it in P.ELECTRONICS_ITEMS:
            mass, behind, up = it[:3]
            side = it[3] if len(it) > 3 else 0.0
            L["pelvis"].items.append(MP.point_props(mass, np.array([-behind * S, side * S, up * S])))
        for size, behind, up, side in getattr(P, "ELECTRONICS_BOXES", []):    # drawn, for the x-ray views
            L["pelvis"].visual.append(("box", (tuple(np.array(size) * MM), (-behind * S, side * S, up * S)), "foot"))
    else:
        L["pelvis"].items.append(MP.point_props(P.ELECTRONICS_KG, np.array(P.ELECTRONICS_POS) * MM))
    _torso(L, org, pieces, lift, waist_motor=P.WAIST)

    for sd, sgn in (("l", 1), ("r", -1)):
        key = f"leg_{sd}"
        lg = to_link_frame(pieces[key], 0.0, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg))
        z_bottom = (legs[sd][2] + dz) * S
        span = max(z_bottom - org[f"ankle_{sd}" if (two or stacked) else f"foot_{sd}"][2], 0.005)
        c_off = legc[sd] - org[key] if hip_pitch else np.zeros(3)
        L[key].items.append(MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.040, span),
                                         (c_off[0], c_off[1], z_bottom - org[key][2] - span / 2)))
        if hip_pitch:                  # bracket: pitch horn down the outside to the leg top
            L[key].items.append(MP.point_props(P.BRACKET_MASS_KG, (c_off[0], c_off[1], c_off[2] / 2)))
        L[key].collision.append(("box", _fit_box(lg.vertices)))
        fkey = f"foot_{sd}"
        if stacked:
            name, sL, RL = feet[sd]
            akey = f"ankle_{sd}"
            rs, rc = NB.case_box_robot(P, extra[sd]["RU"], k)
            _motor(L[key], rs, rc + (org[akey] - org[key]) / MM)              # upper motor in the leg
            fm = to_link_frame(parts[name], 0.0, org[fkey], 0.0)
            L[fkey].mesh_file = fm
            L[fkey].items.append(MP.shell_props(fm))
            L[fkey].items.append(MP.point_props(0.015, (0, 0, -sL[2] * S)))
            _motor(L[fkey], *NB.case_box_robot(P, RL, k))                    # lower motor in the foot
            L[fkey].collision.append(("mesh", fkey))
            L[fkey].z_sole_mm = sL[2] * k
            d = org[fkey] - org[akey]                                         # bracket: one horn to the other
            L[akey].items.append(MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.040, abs(d[2])), d / 2))
            continue
        if two:
            name, sA, sB, RA, RB = feet[sd]
            akey = f"ankle_{sd}"
            fm = to_link_frame(parts[name], 0.0, org[fkey], 0.0)
            L[fkey].mesh_file = fm
            L[fkey].items.append(MP.shell_props(fm))
            L[fkey].items.append(MP.point_props(0.015, (0, 0, -sA[2] * S)))   # sole, screws
            _motor(L[fkey], *NB.case_box_robot(P, RA, k))
            L[fkey].collision.append(("mesh", fkey))
            L[fkey].z_sole_mm = sA[2] * k
            bs, bc = NB.case_box_robot(P, RB, k)
            _motor(L[akey], bs, bc)
            # bracket: pitch motor case onto the roll horn
            d = org[fkey] - org[akey]
            L[akey].items.append(MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.030, 0.020), d / 2))
            continue
        name, th, shaft = feet[sd]
        if slipper:
            ex = extra[sd]
            fm = to_link_frame(trimesh.util.concatenate([ex["cap"], ex["plate"]]), 0.0, org[fkey], 0.0)
            L[fkey].mesh_file = fm
            L[fkey].items.append(MP.shell_props(to_link_frame(ex["cap"], 0.0, org[fkey], 0.0)))
            pl = ex["plate"]
            L[fkey].items.append(MP.box_props(pl.volume * k ** 3 * P.RHO_PLA / 1000.0,   # mm^3 * g/mm^3 -> kg
                                              tuple(pl.extents[[1, 0, 2]] * S),
                                              (-(pl.centroid[1] - shaft[1]) * S, (pl.centroid[0] - shaft[0]) * S,
                                               (pl.centroid[2] - shaft[2]) * S)))
            blo, bhi = ex["box"]
            bc = (blo + bhi) / 2 - shaft
            bhi_case = bhi.copy(); bhi_case[1] -= P.MOTOR_HORN / k           # case without the horn
            ext = (bhi_case - blo) * k                                         # native axes, mm
            size = np.array([ext[1], ext[0], ext[2]])                          # robot axes
            cc = (blo + bhi_case) / 2 - shaft
            centre = np.array([-cc[1], cc[0], cc[2]]) * k
            _motor(L[fkey], size, centre)
            L[fkey].collision.append(("mesh", fkey))
            L[fkey].collision.append(("box", (tuple(size * MM), tuple(centre * MM))))
            L[fkey].z_sole_mm = shaft[2] * k
            # ankle bracket on the leg: a plate down the heel side of the motor to the
            # horn, and a plate under the leg bottom
            z_top = z_bottom - org[key][2]                                     # leg frame
            x_horn = -(bhi[1] - shaft[1]) * S - P.BRACKET_T / 2 * MM            # behind the horn
            y_c = (shaft[0] * S) - org[key][1]
            z_low = org[fkey][2] - 10 * MM - org[key][2]
            x_front = org[fkey][0] - org[key][0] + 12 * MM
            vert = ((P.BRACKET_T * MM, 30 * MM, z_top - z_low), (x_horn - org[key][0] + org[fkey][0], y_c,
                                                                  (z_top + z_low) / 2))
            horiz = ((x_front - (vert[1][0] - P.BRACKET_T / 2 * MM), 30 * MM, P.BRACKET_T * MM),
                     ((x_front + vert[1][0] - P.BRACKET_T / 2 * MM) / 2, y_c, z_top - P.BRACKET_T / 2 * MM))
            for sz, c in (vert, horiz):
                L[key].visual.append(("box", (tuple(sz), tuple(c)), "foot"))
            continue
        fm = to_link_frame(parts[name], 0.0, org[fkey], 0.0)
        L[fkey].mesh_file = fm
        L[fkey].items.append(MP.shell_props(fm))
        L[fkey].items.append(MP.point_props(0.015, (0, 0, -shaft[2] * S)))   # sole plate, screws
        t = np.radians(th)
        d_, h_ = np.array([1.0, 0, 0]), np.array([0, np.sin(t), np.cos(t)])
        R = np.stack([d_, h_, np.cross(d_, h_)], axis=1)
        _motor(L[fkey], np.abs(R) @ np.array([P.MOTOR_D, P.MOTOR_H, P.MOTOR_W]),
               R @ np.array([0.0, LY.H_MID, 0.0]))
        L[fkey].collision.append(("mesh", fkey))
        L[fkey].z_sole_mm = shaft[2] * k

    if getattr(P, "LINERS", False):
        # holes in the STL surface: line the inside in the shell colour (visual only,
        # added after the shell masses were taken)
        body = sol["body_legs"]
        shares = {"pelvis": (body, lambda q: (q[:, 2] >= P.CUT_LEG_TOP) & (q[:, 2] <= P.CUT_WAIST), lift),
                  "torso": (body, lambda q: q[:, 2] >= P.CUT_WAIST, lift)}
        for sd, sgn in (("l", 1), ("r", -1)):
            shares[f"leg_{sd}"] = (body, lambda q, sgn=sgn: (q[:, 2] <= P.CUT_LEG_TOP) & (sgn * q[:, 0] > 0), lift)
            shares[f"foot_{sd}"] = (sol[feet[sd][0]], None, 0.0)
        if seam is not None:
            bs = seam["belly_solid"]
            shares["pelvis"] = (body, lambda q: bs.inside(q) & (q[:, 2] <= P.CUT_WAIST), lift)
            shares["torso"] = (body, lambda q: bs.inside(q) & (q[:, 2] >= P.CUT_WAIST), lift)
            for sd in ("l", "r"):
                shares[f"leg_{sd}"] = ((seam["leg_solid"][sd], None, lift) if seam.get("uniform")
                                       else (body, seam["leg_solid"][sd].inside, lift))
        for key, (solid_, keep, lift_) in shares.items():
            ln = NB.liner(solid_, keep)
            if ln is not None:
                L[key].mesh_file = trimesh.util.concatenate(
                    [L[key].mesh_file, to_link_frame(ln, 0.0, org[key], lift_)])

    # display: where make_display_model put it for this character size
    meta_display = np.array([-dc[1] * S, dc[0] * S, dc[2] * S + lift * MM])

    # motor containment, against the voxel solids, in the STL frame
    solids = list(sol.values())
    inside, inside_w = {}, {}
    for n, l in L.items():
        lifted = not n.startswith(("foot", "ankle"))
        if slipper and not lifted:
            solids_n = [extra[n[-1]]["cap_solid"]]
        elif (two or stacked) and not lifted:
            solids_n = [sol[feet[n[-1]][0]]]
        elif stacked and n.startswith("leg"):
            ex = extra[n[-1]]
            solids_n = [NB.union_solid(((ex["poly"].bounds[0], ex["poly"].bounds[1], legs[n[-1]][2]),
                                        (ex["poly"].bounds[2], ex["poly"].bounds[3], P.CUT_LEG_TOP)),
                                       [lambda q, ex=ex, sd=n[-1]: contains_xy(ex["poly"], q[:, 0], q[:, 1])
                                        & (q[:, 2] >= legs[sd][2]) & (q[:, 2] <= P.CUT_LEG_TOP)])] + solids
        else:
            solids_n = solids
        for i, (size, c) in enumerate(l.motors):
            w = c + org[n]
            cn = np.array([w[1] / S, -w[0] / S, (w[2] - (lift * MM if lifted else 0.0)) / S])
            sn = np.array([size[1], size[0], size[2]]) / S
            inside[f"{n}#{i}"] = NB.motor_fraction_inside(solids_n, sn, cn)
            inside_w[f"{n}#{i}"] = NB.motor_fraction_inside(solids_n, sn + 2 * P.WALL / k, cn)

    joints = [("waist_yaw", "pelvis", "torso", (0, 0, 1), org["torso"] - org["pelvis"], "waist_yaw")
              if P.WAIST else ("waist_fixed", "pelvis", "torso", (0, 0, 1), org["torso"] - org["pelvis"], None)]
    for sd in ("l", "r"):
        if hip_pitch and not hip_yaw:
            joints.append((f"hip_pitch_{sd}", "pelvis", f"leg_{sd}", (0, 1, 0),
                           org[f"leg_{sd}"] - org["pelvis"], "hip_pitch"))
        elif hip_pitch:
            joints.append((f"hip_yaw_{sd}", "pelvis", f"hip_{sd}", (0, 0, 1),
                           org[f"hip_{sd}"] - org["pelvis"], "hip_yaw"))
            joints.append((f"hip_pitch_{sd}", f"hip_{sd}", f"leg_{sd}", (0, 1, 0),
                           org[f"leg_{sd}"] - org[f"hip_{sd}"], "hip_pitch"))
        else:
            joints.append((f"hip_yaw_{sd}", "pelvis", f"leg_{sd}", (0, 0, 1),
                           org[f"leg_{sd}"] - org["pelvis"], "hip_yaw"))
    if stacked:
        ax = {"roll": (1, 0, 0), "pitch": (0, 1, 0)}
        upj = P.STACK_UP
        lowj = "pitch" if upj == "roll" else "roll"
        for sd in ("l", "r"):
            joints.append((f"ankle_{upj}_{sd}", f"leg_{sd}", f"ankle_{sd}", ax[upj],
                           org[f"ankle_{sd}"] - org[f"leg_{sd}"], f"ankle_{upj}"))
        for sd in ("l", "r"):
            joints.append((f"ankle_{lowj}_{sd}", f"ankle_{sd}", f"foot_{sd}", ax[lowj],
                           org[f"foot_{sd}"] - org[f"ankle_{sd}"], f"ankle_{lowj}"))
    elif two:
        for sd in ("l", "r"):
            joints.append((f"ankle_pitch_{sd}", f"leg_{sd}", f"ankle_{sd}", (0, 1, 0),
                           org[f"ankle_{sd}"] - org[f"leg_{sd}"], "ankle_pitch"))
        for sd in ("l", "r"):
            joints.append((f"ankle_roll_{sd}", f"ankle_{sd}", f"foot_{sd}", (1, 0, 0),
                           org[f"foot_{sd}"] - org[f"ankle_{sd}"], "ankle_roll"))
    else:
        for sd in ("l", "r"):
            joints.append((f"ankle_roll_{sd}", f"leg_{sd}", f"foot_{sd}", (1, 0, 0),
                           org[f"foot_{sd}"] - org[f"leg_{sd}"], "ankle_roll"))
    fe = parts["foot_a"].extents * k
    sole = parts["foot_a"].vertices[parts["foot_a"].vertices[:, 2] < 1e-6]
    if stacked:                                  # roll axis: upper (in the leg) or lower (in the foot)
        if P.STACK_UP == "roll":
            sl, sr = extra["l"]["up_pt"] + [0, 0, dz], extra["r"]["up_pt"] + [0, 0, dz]
        else:
            sl, sr = feet["l"][1], feet["r"][1]
    else:
        sl, sr = (feet["l"][1], feet["r"][1]) if two else (feet["l"][2], feet["r"][2])
    n_ank = "ankle pitch x2, ankle roll x2" if (two or stacked) else "ankle roll x2"
    n_hip = ("hip yaw x2, hip pitch x2" if hip_yaw else "hip pitch x2") if hip_pitch else "hip yaw x2"
    n_act = 4 + 2 * (two or stacked) + 2 * hip_pitch - 2 * (not hip_yaw)
    summary = {"actuated joints": (f"{n_act + 1} (waist yaw, {n_hip}, {n_ank})" if P.WAIST
                                   else f"{n_act} ({n_hip}, {n_ank}; the waist is fixed)"),
               "servo": f"{P.SERVO}: {P.MOTOR_W} x {P.MOTOR_H} x {P.MOTOR_D} mm, {P.MOTOR_MASS_KG*1000:.0f} g, "
                        f"stall {P.MOTOR_STALL_NM} N.m",
               "roll axis above sole (mm)": f"{sl[2]*k:.1f} / {sr[2]*k:.1f} (left / right)",
               "roll axis vs leg centre (mm)": f"{(legs['l'][1]-sl[1])*k:.1f} ahead, "
                                               f"{(legs['l'][0]-sl[0])*k:.1f} inboard",
               "legs raised to clear the feet (mm)": round(lift, 1)}
    if hip_pitch:
        sh, _, a_, b_ = hp["l"]
        summary["hip pitch motor"] = (f"standing in the belly" + (" under the hip yaw motor" if hip_yaw else "")
                                      + f", shaft lateral, horn outboard; "
                                      f"axis {(sh[2] - P.CUT_LEG_TOP) * k:.1f} mm above the crotch, "
                                      f"{a_:.1f} ahead of the leg centre (target {float(P.HIP_PITCH_AHEAD):.1f}), "
                                      f"case {b_:+.1f} outboard; walled case inside the body"
                                      + (f" over hip yaw +-{P.LIM['hip_yaw'][1]:.0f} deg" if hip_yaw else ""))
        if hip_yaw:
            summary["hip yaw motor"] = f"vertical shaft, horn down, case bottom {P.HIP_YAW_LIFT:.1f} mm above the crotch"
    if slipper:
        ex = extra["l"]
        ctop = ex["box"][1][2] * k
        summary.update({
            "character scale": f"x{k:.3f} of the 120 mm STL ({P.CHAR_HEIGHT:.0f} mm character)",
            "foot (mm)": f"{fe[1]:.1f} long x {fe[0]:.1f} wide (the STL's outline); toe cap "
                         f"{(P.SLIPPER_CUT_Y - fb[0][1]) * k:.0f} mm long, sole plate {P.SOLE_PLATE:.0f} mm",
            "roll motor": f"{P.ANKLE_MOTOR} on the sole plate at the heel, in the open, shaft fore-aft, "
                          + ("long side up" if P.ANKLE_MOTOR == "standing" else "long end outward")
                          + ", horn to the heel",
            "ankle gap, motor top to leg cap (mm)": round((min(legs['l'][2], legs['r'][2]) + dz) * k - ctop, 1),
            "leg shell (mm)": f"original, capped at {P.LEG_CUT_Z * k:.0f} mm, "
                              f"{(P.CUT_LEG_TOP - P.LEG_CUT_Z) * k:.1f} from cap to crotch"})
    elif stacked:
        gap = (min(legs["l"][2], legs["r"][2]) + dz - max(parts[n].bounds[1][2] for n in ("foot_a", "foot_b"))) * k
        sPl = feet["l"][1] if P.STACK_UP == "roll" else extra["l"]["up_pt"] + [0, 0, dz]
        summary.update({
            "character scale": f"x{k:.2f} of the 120 mm STL ({P.CHAR_HEIGHT:.0f} mm character)",
            "foot (mm)": f"{fe[1]:.1f} long x {fe[0]:.1f} wide x {fe[2]:.1f} high (the STL's own foot, "
                         f"flat sole {np.ptp(sole[:, 1]) * k:.0f} x {np.ptp(sole[:, 0]) * k:.0f})",
            "ankle motors": ("pitch lying in the foot (long side to the toe); roll standing in the leg above it"
                             if P.STACK_UP == "roll" else
                             "roll lying in the foot (long end outward); pitch standing in the leg above it"),
            "pitch axis above sole (mm)": f"{sPl[2]*k:.1f}, {(legs['l'][1]-sPl[1])*k:.1f} ahead of the leg centre",
            "leg (mm)": f"the STL leg lengthened by its own section: {(P.LEG_CUT_Z - legs['l'][2]) * k:.0f} added, "
                        f"{(P.CUT_LEG_TOP - legs['l'][2]) * k:.0f} from bottom to crotch"
                        + (f"; the added part reaches {extra['l']['front_ext'] * k:.1f} further forward"
                           if extra["l"].get("front_ext") else ""),
            "ankle gap, leg bottom to foot top (mm)": round(gap, 1)})
    elif two:
        gap = (min(legs["l"][2], legs["r"][2]) + dz - max(parts[n].bounds[1][2] for n in ("foot_a", "foot_b"))) * k
        sBl = feet["l"][2]
        summary.update({
            "character scale": f"x{k:.2f} of the 120 mm STL ({P.CHAR_HEIGHT:.0f} mm character)",
            "foot (mm)": f"{fe[1]:.1f} long x {fe[0]:.1f} wide x {fe[2]:.1f} high (the STL's own foot, "
                         f"flat sole {np.ptp(sole[:, 1]) * k:.0f} x {np.ptp(sole[:, 0]) * k:.0f})",
            "ankle motors": f"both inside the foot: roll standing, pitch lying with its long side "
                            f"to the {P.PITCH_LONG_END}, riding the roll output",
            "pitch axis above sole (mm)": f"{sBl[2]*k:.1f}, {(legs['l'][1]-sBl[1])*k:.1f} ahead of the leg centre",
            "ankle gap, leg bottom to foot top (mm)": round(gap, 1),
            "leg shell (mm)": f"original, {(P.CUT_LEG_TOP - min(legs['l'][2], legs['r'][2])) * k:.1f} from bottom to crotch"})
    else:
        gap = (min(legs["l"][2], legs["r"][2]) + dz - max(parts[n].bounds[1][2] for n in ("foot_a", "foot_b"))) * k
        summary.update({
            "character scale": f"x{k:.2f} of the 120 mm STL ({P.CHAR_HEIGHT:.0f} mm character; "
                               f"the walled roll motor fits the STL foot from x{NB.min_scale(P, sol['foot_a'], P.FOOT_MOTOR_THETA):.2f})",
            "foot (mm)": f"{fe[1]:.1f} long x {fe[0]:.1f} wide x {fe[2]:.1f} high (the STL's own foot, "
                         f"flat sole {np.ptp(sole[:, 1]) * k:.0f} x {np.ptp(sole[:, 0]) * k:.0f})",
            "roll motor": "lying on its side in the foot, shaft fore-aft, long end outward",
            "ankle gap, leg bottom to foot top (mm)": round(gap, 1),
            "leg shell (mm)": f"original, {(P.CUT_LEG_TOP - min(legs['l'][2], legs['r'][2])) * k:.1f} from bottom to crotch"})
    excl = [("leg_l", "foot_l"), ("leg_r", "foot_r")]
    if two or stacked:
        excl += [("leg_l", "ankle_l"), ("leg_r", "ankle_r"), ("ankle_l", "foot_l"), ("ankle_r", "foot_r")]
    if hip_pitch:
        # MuJoCo's convex shapes cannot follow the leg / belly seam (they would touch all the
        # time); that pair is checked on the real part volumes instead (hip_range in main),
        # and the hip joint limits are set to the collision-free range.  All other pairs stay.
        excl += [("pelvis", "leg_l"), ("pelvis", "leg_r"), ("torso", "leg_l"), ("torso", "leg_r")]
    meta = dict(org=org, heights=hz, exclude=excl, leg_ref_x=legc["l"][0],
                display_world=meta_display, inside=inside, inside_walled=inside_w, summary=summary)
    if seam is not None:
        sh = lambda s_: NB.Solid(s_.grid, s_.lo - [0, 0, dz], s_.pitch)      # feet into the body's frame
        t_, w_, h_ = np.array(P.DISPLAY_OUTLINE) / k
        dlo = np.array([dc[0] - w_ / 2, dc[1] - t_ / 2, dc[2] - h_ / 2])
        dhi = np.array([dc[0] + w_ / 2, dc[1] + t_ / 2, dc[2] + h_ / 2])
        xp = {}
        if stacked:                    # the added leg below the STL, for the hip range check
            for sd in ("l", "r"):
                xp[sd], _ = trimesh.sample.sample_surface(extra[sd]["prism"], 4000, seed=5)
        meta["seam"] = dict(R=seam["R"], k=k, centres=seam["centres"], legs=seam["leg_solid"], extra_pts=xp,
                            belly=seam["belly_solid"], backpack=sol["backpack"],
                            feet={sd: sh(sol[feet[sd][0]]) for sd in ("l", "r")}, display=(dlo, dhi))
        if "z_split" in seam:
            sg_ = seam["socket_gap"]["l"]
            summary["hip seam"] = (f"torso cut flat {(seam['z_split'] - P.CUT_LEG_TOP) * k:.1f} mm above the STL crotch "
                                   f"({getattr(P, 'HIP_SOCKET_AT', 'pivot')}), legs carried up into sockets (legs that "
                                   f"much longer), hidden dome R {seam['r_dome']['l'] * k:.1f} mm about the hip pivot; "
                                   f"sockets cut for hip pitch {P.HIP_CARVE_PITCH[0]:.0f}..{P.HIP_CARVE_PITCH[1]:.0f}, "
                                   f"yaw +-{P.HIP_CARVE_YAW:.0f} + {P.HIP_SEAM_GAP:.1f} mm gap: socket gap front "
                                   f"{sg_['front']:.1f} / back {sg_['back']:.1f} / outer {sg_['outer']:.1f} mm, "
                                   f"{'BREAKS the torso skin' if sg_['broken'] else 'torso wall kept'} "
                                   f"(thickest wall beside it {sg_['wall']:.1f} mm); "
                                   f"motor hollow through the visible leg skin {seam['slot_skin']['l']:.0f} mm^2; "
                                   f"leg points reaching the hip yaw case {seam['motor_hit']}")
        elif seam["R"] is not None:
            summary["hip seam"] = (f"outside: sphere R {seam['R'] * k:.1f} mm about the point where hip yaw and pitch "
                                   f"meet; between the legs: the crotch plane; {P.HIP_SEAM_GAP:.1f} mm gap; "
                                   f"legs and belly keep the STL's outside")
        else:
            summary["hip seam"] = (f"legs cut off whole at the crotch (flat lid); lower belly carved around their sweep "
                                   f"(hip pitch {P.HIP_CARVE_PITCH[0]:.0f}..{P.HIP_CARVE_PITCH[1]:.0f} deg, yaw "
                                   f"+-{P.HIP_CARVE_YAW:.0f}) + {P.HIP_SEAM_GAP:.1f} mm gap: pockets up to "
                                   f"{seam['carve_depth'] * k:.1f} mm above the crotch; leg points reaching into "
                                   f"the hip motor cases: {seam['motor_hit']}")
    return L, joints, meta


BUILDERS = {"exposed": build_exposed, "foot_both": build_foot_both, "split": build_split,
            "roll_foot": build_roll_foot, "newbase": build_newbase}


def add_display(L, meta):
    """Put the 9 inch display in the fists, screen facing forward, rigid with
    the torso.  Centred in the fists' thickness and width; lowered from the fist
    centre until it clears the body and the chin by DISPLAY_CLEAR."""
    org = meta["org"]
    t, w, h = np.array(P.DISPLAY_OUTLINE) * MM
    if "display_world" in meta:                                  # placed with the character's arms
        cx, _, cz = meta["display_world"]
    else:
        tr = L["torso"].mesh_file.vertices + org["torso"]        # world, metres
        arm = tr[np.abs(tr[:, 1]) > 80 * MM]
        arm = arm[arm[:, 2] < org["torso"][2] + 90 * MM]
        fist = arm[arm[:, 0] > arm[:, 0].max() - 25 * MM]
        cx = (fist[:, 0].min() + fist[:, 0].max()) / 2.0
        cz = (fist[:, 2].min() + fist[:, 2].max()) / 2.0
    body = np.vstack([L[k].mesh_file.vertices + org[k] for k in ("torso", "pelvis")])
    body = body[np.abs(body[:, 1]) < 80 * MM]                    # everything but the arms
    g = P.DISPLAY_CLEAR * MM
    while "display_world" not in meta:
        inside = ((np.abs(body[:, 0] - cx) < t / 2 + g) & (np.abs(body[:, 1]) < w / 2 + g)
                  & (np.abs(body[:, 2] - cz) < h / 2 + g))
        if not inside.any():
            break
        cz -= 0.5 * MM
    c = np.array([cx, 0.0, cz]) - org["torso"]                   # torso frame
    L["torso"].items.append(MP.box_props(P.DISPLAY_MASS_KG + P.DISPLAY_MOUNT_KG, (t, w, h), c))
    L["torso"].visual.append(("box", ((t, w, h), tuple(c)), "display"))
    aw, ah = np.array(P.DISPLAY_ACTIVE) * MM
    L["torso"].visual.append(("box", ((0.4 * MM, aw, ah), tuple(c + [t / 2, 0, 0])), "screen"))
    L["torso"].collision.append(("box", ((t, w, h), tuple(c))))
    meta["summary"]["display"] = (f"{P.DISPLAY_OUTLINE[1]:.1f} x {P.DISPLAY_OUTLINE[2]:.1f} x "
                                  f"{P.DISPLAY_OUTLINE[0]:.1f} mm, {P.DISPLAY_MASS_KG + P.DISPLAY_MOUNT_KG:.3f} kg, "
                                  f"centre {cx*1000:.0f} mm ahead, {cz*1000:.0f} mm up, "
                                  f"bottom edge {(cz - h/2)*1000:.0f} mm above the floor")


def motors_inside(L, org, n=6):
    """Fraction of each motor case inside the character's shells (neutral pose)."""
    world = []
    for k, l in L.items():
        if l.mesh_file is not None:
            m = l.mesh_file.copy()
            m.apply_translation(org[k])
            world.append(m)
    g = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    out = {}
    for k, l in L.items():
        for i, (size, c) in enumerate(l.motors):
            pts = grid * size + c + org[k]
            inside = np.zeros(len(pts), bool)
            for m in world:
                lo, hi = m.bounds
                near = ~inside & np.all((pts >= lo) & (pts <= hi), axis=1)
                if near.any():
                    inside[near] |= m.contains(pts[near])
            out[f"{k}#{i}"] = float(inside.mean())
    return out


# --------------------------------------------------------------------------- #
#  URDF
# --------------------------------------------------------------------------- #
RGBA = {
    "display": "0.18 0.19 0.21 1",       # metal case / bezel
    "screen": "0.62 0.80 0.95 1",
    "shell":  "0.106 0.216 0.541 1",     # Haetburi blue
    "motor":  "0.15 0.15 0.17 1",
    "foot":   "0.25 0.25 0.28 1",
}


def f3(v):
    return " ".join(f"{x:.6g}" for x in v)


def urdf_inertial(link):
    m, c, I = link.props()
    return (f'    <inertial>\n'
            f'      <origin xyz="{f3(c)}" rpy="0 0 0"/>\n'
            f'      <mass value="{m:.6g}"/>\n'
            f'      <inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}"'
            f' iyy="{I[1,1]:.6g}" iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/>\n'
            f'    </inertial>\n')


def write_urdf(L, joints, path):
    out = ['<?xml version="1.0"?>',
           f'<!-- Haetburi biped ({P.VARIANT}) - generated by tools/build_model.py, do not edit by hand -->',
           '<robot name="haetburi">',
           '  <mujoco>',
           '    <compiler meshdir="." balanceinertia="true" discardvisual="false"'
           ' strippath="false" fusestatic="false"/>',
           '  </mujoco>']
    for name, rgba in RGBA.items():
        out.append(f'  <material name="{name}"><color rgba="{rgba}"/></material>')

    out.append('  <link name="world"/>')
    out.append('  <joint name="floating_base" type="floating">')
    out.append('    <parent link="world"/><child link="pelvis"/>')
    out.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
    out.append('  </joint>')

    for name, link in L.items():
        out.append(f'  <link name="{name}">')
        out.append(urdf_inertial(link).rstrip('\n'))
        if link.mesh_file is not None:
            out.append(f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
                       f'<geometry><mesh filename="meshes/{name}.stl"/></geometry>'
                       f'<material name="shell"/></visual>')
        for kind, params, mat in link.visual:
            size, pos = params
            out.append(f'    <visual><origin xyz="{f3(pos)}" rpy="0 0 0"/>'
                       f'<geometry><box size="{f3(size)}"/></geometry>'
                       f'<material name="{mat}"/></visual>')
        for kind, params in link.collision:
            if kind == "mesh":
                pos = (0, 0, 0)
                g = f'<mesh filename="meshes/{params}.stl"/>'
            elif kind == "box":
                size, pos = params
                g = f'<box size="{f3(size)}"/>'
            else:
                r, pos = params
                g = f'<sphere radius="{r:.6g}"/>'
            out.append(f'    <collision><origin xyz="{f3(pos)}" rpy="0 0 0"/>'
                       f'<geometry>{g}</geometry></collision>')
        out.append('  </link>')

    for jn, parent, child, axis, xyz, lk in joints:
        if lk is None:                                 # fixed (no motor)
            out.append(f'  <joint name="{jn}" type="fixed">')
            out.append(f'    <parent link="{parent}"/><child link="{child}"/>')
            out.append(f'    <origin xyz="{f3(xyz)}" rpy="0 0 0"/>')
            out.append('  </joint>')
            continue
        lo, hi = P.LIM[lk]
        out.append(f'  <joint name="{jn}" type="revolute">')
        out.append(f'    <parent link="{parent}"/><child link="{child}"/>')
        out.append(f'    <origin xyz="{f3(xyz)}" rpy="0 0 0"/>')
        out.append(f'    <axis xyz="{f3(axis)}"/>')
        out.append(f'    <limit lower="{lo*DEG:.6g}" upper="{hi*DEG:.6g}"'
                   f' effort="{P.MOTOR_STALL_NM}" velocity="{P.MOTOR_NOLOAD_RADS}"/>')
        out.append(f'    <dynamics damping="{P.JOINT_DAMPING}" friction="{P.JOINT_FRICTION}"/>')
        out.append('  </joint>')

    out.append('</robot>')
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def hip_range(sm, margin=2.0):
    import newbase as NB
    """Collision-free hip range on the real part volumes (voxel solids): the left leg
    and foot turned about its hip against the belly (motors inside), backpack, the
    other leg and foot, and the display.  Per joint with the other at zero, and a
    yaw x pitch map (robot/<variant>/hip_range.json).  Sets P.LIM to that range less
    `margin` (never wider than designed)."""
    c = sm["centres"]["l"]
    moving = np.vstack([NB.boundary_points(sm["legs"]["l"], 2), NB.boundary_points(sm["feet"]["l"], 2)]
                       + ([sm["extra_pts"]["l"]] if sm.get("extra_pts") else []))
    dlo, dhi = sm["display"]
    obst = {"belly": sm["belly"].inside, "backpack": sm["backpack"].inside,
            "other leg": sm["legs"]["r"].inside, "other foot": sm["feet"]["r"].inside,
            "display": NB.in_box(dlo, dhi)}
    base = NB.hip_collisions(moving, obst, c, 0, 0)

    def hit(yaw, pitch):
        h = NB.hip_collisions(moving, obst, c, yaw, pitch)
        return [n for n in h if h[n] > base[n] + 3]

    def sweep(axis, sign, top):
        for a in range(1, top + 1):
            h = hit(sign * a, 0) if axis == "yaw" else hit(0, sign * a)
            if h:
                return a - 1, h
        return top, []
    fwd, fwd_by = sweep("pitch", -1, 90)
    back, back_by = sweep("pitch", 1, 90)
    out_, out_by = sweep("yaw", 1, 60)          # left leg: + yaw turns the toes outward
    in_, in_by = sweep("yaw", -1, 60)
    yaws, pitches = list(range(-30, 31, 5)), list(range(-90, 91, 2))
    grid = [[hit(y, p_) for p_ in pitches] for y in yaws]
    with open(os.path.join(P.OUT_DIR, "hip_range.json"), "w") as fh:
        json.dump(dict(yaw=yaws, pitch=pitches, hits=grid, fwd=fwd, back=back, yaw_out=out_, yaw_in=in_,
                       by=dict(fwd=fwd_by, back=back_by, out=out_by, inn=in_by), neutral=base), fh)
    lo0, hi0 = P.LIM["hip_pitch"]
    ylo0, yhi0 = P.LIM["hip_yaw"]
    P.LIM = dict(P.LIM, hip_pitch=(max(lo0, -(fwd - margin)), min(hi0, back - margin)))
    ym = min(yhi0, in_ - margin, out_ - margin)
    P.LIM["hip_yaw"] = (-ym, ym)
    why = lambda b: ", ".join(b) if b else "not reached"
    warn = {n: v for n, v in base.items() if v > 50}
    return {**({"WARNING, parts overlapping in the standing pose (points)": warn} if warn else {}),
            "hip range, collision-free (deg)":
            f"pitch {fwd} forward ({why(fwd_by)}) / {back} back ({why(back_by)}); "
            f"yaw {in_} toes-in ({why(in_by)}) / {out_} toes-out ({why(out_by)})",
            "hip joint limits set (deg)": f"pitch {P.LIM['hip_pitch'][0]:.0f}..{P.LIM['hip_pitch'][1]:.0f} "
                                          f"(+ = leg back), yaw +-{ym:.0f}"}


def main():
    os.makedirs(P.MESH_DIR, exist_ok=True)
    for old in os.listdir(P.MESH_DIR):                 # this variant's folder only
        os.remove(os.path.join(P.MESH_DIR, old))
    def assemble():
        L, joints, meta = BUILDERS[P.LAYOUT]()
        if P.DISPLAY:
            add_display(L, meta)
        return L, joints, meta

    ank = getattr(P, "PITCH_AXIS_AHEAD_NATIVE", None) == "auto"
    hipa = getattr(P, "HIP_PITCH_AHEAD", None) == "auto"
    if ank or hipa:
        # the ankle pitch axis under the robot's CoM and / or the hip pitch axis under
        # the CoM of what the hips carry: build, measure, rebuild (both targets together)
        if ank:
            P.PITCH_AXIS_AHEAD_NATIVE = 1.26
        if hipa:
            P.HIP_PITCH_AHEAD = 0.0
        for _ in range(2):
            L, joints, meta = assemble()
            ref = meta.get("leg_ref_x", meta["org"]["leg_l"][0])          # leg centre
            if ank:
                items = [(l.props()[0], l.props()[1] + meta["org"][n], l.props()[2]) for n, l in L.items()]
                P.PITCH_AXIS_AHEAD_NATIVE = round((MP.combine(items)[1][0] - ref) / S, 2)
            if hipa:
                items = [(L[n].props()[0], L[n].props()[1] + meta["org"][n], L[n].props()[2])
                         for n in ("pelvis", "torso", "hip_l", "hip_r") if n in L]
                P.HIP_PITCH_AHEAD = round((MP.combine(items)[1][0] - ref) / MM, 1)
        if ank:
            print(f"ankle pitch axis target: {P.PITCH_AXIS_AHEAD_NATIVE * P.MESH_SCALE:.1f} mm ahead of the leg centre (the CoM)")
        if hipa:
            print(f"hip pitch axis target: {P.HIP_PITCH_AHEAD:.1f} mm ahead of the leg centre (upper-body CoM)")
    if P.PITCH_AXIS_X == "auto":
        # put the ankle pitch axis under the whole robot's CoM (two passes: moving
        # the foot moves the CoM a little)
        P.PITCH_AXIS_X = 0.0
        for _ in range(2):
            L, joints, meta = assemble()
            items = [(l.props()[0], l.props()[1] + meta["org"][n], l.props()[2]) for n, l in L.items()]
            com_x = MP.combine(items)[1][0]
            leg_x = meta["org"]["leg_l"][0]
            P.PITCH_AXIS_X = round((com_x - leg_x) / MM + float(P.PITCH_AXIS_X)
                                   - (meta["org"]["foot_l"][0] - leg_x) / MM, 1)
    L, joints, meta = assemble()
    if "seam" in meta:
        meta["summary"].update(hip_range(meta["seam"]))
    if "inside" not in meta:
        meta["inside"] = motors_inside(L, meta["org"])
    for name, link in L.items():
        if link.mesh_file is not None:
            if len(link.mesh_file.faces) > 190000:      # MuJoCo reads at most 200k faces a mesh
                link.mesh_file = link.mesh_file.simplify_quadric_decimation(face_count=190000)
            link.mesh_file.export(os.path.join(P.MESH_DIR, f"{name}.stl"))
    urdf = os.path.join(P.OUT_DIR, "haetburi.urdf")
    write_urdf(L, joints, urdf)
    scene = os.path.join(P.OUT_DIR, "haetburi_scene.xml")
    write_mjcf.write(L, joints, meta, scene)

    items = [(l.props()[0], l.props()[1] + meta["org"][n], l.props()[2]) for n, l in L.items()]
    M, C, _ = MP.combine(items)
    hz = meta["heights"]
    lines = [f"variant  {P.VARIANT}: {P.TITLE}",
             f"wrote    {os.path.relpath(urdf, P.ROOT)}, {os.path.relpath(scene, P.ROOT)}", "",
             f"{'link':9s} {'mass kg':>8s}  com (m, robot frame)"]
    for n, l in L.items():
        m, c, _ = l.props()
        lines.append(f"{n:9s} {m:8.3f}  {np.round(c + meta['org'][n], 4)}")
    lines += ["", f"total mass {M:.3f} kg   com {np.round(C, 4)} m   com height {C[2]*1000:.0f} mm",
              f"standing height {hz['top']:.0f} mm, character scale x{P.MESH_SCALE:.4f}"]
    lines += [f"{k}: {v}" for k, v in meta["summary"].items()]
    iw = meta.get("inside_walled", {})
    lines.append("motor volume inside the shell" + (f" (bare case / with its {P.WALL:.1f} mm walls):" if iw else ":"))
    lines += [f"  {k:10s} {v*100:5.1f} %" + (f" / {iw[k]*100:5.1f} %" if k in iw else "")
              for k, v in meta["inside"].items()]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(P.OUT_DIR, "build_report.txt"), "w") as fh:
        fh.write(report + "\n")
    info = dict(variant=P.VARIANT, title=P.TITLE, layout=P.LAYOUT, demo=P.DEMO,
                mass_kg=round(M, 3), com_mm=[round(v * 1000, 1) for v in C],
                height_mm=round(hz["top"], 1), summary=meta["summary"],
                motors_inside={k: round(v, 3) for k, v in meta["inside"].items()})
    with open(os.path.join(P.OUT_DIR, "variant.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    return L, joints, meta


if __name__ == "__main__":
    main()
