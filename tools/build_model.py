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


BUILDERS = {"exposed": build_exposed, "foot_both": build_foot_both, "split": build_split,
            "roll_foot": build_roll_foot}


def add_display(L, meta):
    """Put the 9 inch display in the fists, screen facing forward, rigid with
    the torso.  Centred in the fists' thickness and width; lowered from the fist
    centre until it clears the body and the chin by DISPLAY_CLEAR."""
    org = meta["org"]
    tr = L["torso"].mesh_file.vertices + org["torso"]            # world, metres
    arm = tr[np.abs(tr[:, 1]) > 80 * MM]
    arm = arm[arm[:, 2] < org["torso"][2] + 90 * MM]
    fist = arm[arm[:, 0] > arm[:, 0].max() - 25 * MM]
    t, w, h = np.array(P.DISPLAY_OUTLINE) * MM
    cx = (fist[:, 0].min() + fist[:, 0].max()) / 2.0
    cz = (fist[:, 2].min() + fist[:, 2].max()) / 2.0
    body = np.vstack([L[k].mesh_file.vertices + org[k] for k in ("torso", "pelvis")])
    body = body[np.abs(body[:, 1]) < 80 * MM]                    # everything but the arms
    g = P.DISPLAY_CLEAR * MM
    while True:
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


def main():
    os.makedirs(P.MESH_DIR, exist_ok=True)
    for old in os.listdir(P.MESH_DIR):                 # this variant's folder only
        os.remove(os.path.join(P.MESH_DIR, old))
    def assemble():
        L, joints, meta = BUILDERS[P.LAYOUT]()
        if P.DISPLAY:
            add_display(L, meta)
        return L, joints, meta

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
    meta["inside"] = motors_inside(L, meta["org"])
    for name, link in L.items():
        if link.mesh_file is not None:
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
    lines.append("motor volume inside the shell:")
    lines += [f"  {k:10s} {v*100:5.1f} %" for k, v in meta["inside"].items()]
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
