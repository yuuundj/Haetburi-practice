#!/usr/bin/env python3
"""Build the Haetburi biped: link meshes, URDF, MJCF scene.

Run:  python3 tools/build_model.py
Out:  robot/meshes/*.stl, robot/haetburi.urdf, robot/haetburi_scene.xml
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import haetburi_params as P
import massprops as MP
from segment import segment
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
        self.visual = []         # ('mesh'|'box'|'cylinder', params, rgba)
        self.collision = []      # ('box'|'sphere', params)
        self.mesh_file = None

    def props(self):
        return MP.combine(self.items)


def _motor(link, size_mm, centre_mm):
    size, c = np.asarray(size_mm) * MM, np.asarray(centre_mm) * MM
    link.items.append(MP.box_props(P.MOTOR_MASS_KG, size, c))
    link.visual.append(("box", (tuple(size), tuple(c)), "motor"))
    link.motors.append((size, c))


def build():
    # ---- sizes that do not depend on where the body ends up
    fl = LY.foot_layout()
    f, z_p = fl["foot"], fl["z_pitch"]                   # pitch axis height above the sole
    leg_half = LY.leg_section()
    leg_len, roll_above_bottom, _ = LY.leg_length()
    pieces = segment(leg_len, leg_half)

    # ---- where the legs are
    c_l = pieces["leg_l"].metadata["centre_native"]
    c_r = pieces["leg_r"].metadata["centre_native"]
    assert abs((c_l[0] - c_r[0]) / 2.0 - P.LEG_HALF_SPACING_NATIVE) < 0.3, \
        "update LEG_HALF_SPACING_NATIVE in haetburi_params.py"
    y_hip_mm = (c_l[0] - c_r[0]) / 2.0 * P.MESH_SCALE
    x_axis = -(c_l[1] + c_r[1]) / 2.0 * S

    # ---- float the leg just clear of its foot through the whole ankle range
    bottom_native = pieces["leg_l"].metadata["bottom_native"]
    leg0 = to_link_frame(pieces["leg_l"], c_l[0] - y_hip_mm / P.MESH_SCALE,
                         (x_axis, y_hip_mm * MM, 0.0), -bottom_native * P.MESH_SCALE)
    leg_rel = leg0.vertices / MM
    leg_rel = leg_rel[leg_rel[:, 2] < 30.0]
    z_leg_bottom = z_p + LY.leg_clear_height(f, leg_rel, roll_above_bottom)
    z_r = z_leg_bottom + roll_above_bottom
    hz = LY.body_heights(z_leg_bottom, leg_len)
    lift = hz["lift"]

    org = {
        "pelvis": np.array([x_axis, 0.0, hz["hip"] * MM]),
        "torso":  np.array([x_axis, 0.0, hz["waist"] * MM]),
    }
    for side, sgn in (("l", 1), ("r", -1)):
        y = sgn * y_hip_mm * MM
        org[f"leg_{side}"] = np.array([x_axis, y, hz["hip"] * MM])
        org[f"ankle_{side}"] = np.array([x_axis, y, z_r * MM])
        org[f"foot_{side}"] = np.array([x_axis, y, z_p * MM])

    L = {n: Link(n) for n in org}
    for l in L.values():
        l.motors = []

    # ---- pelvis: belly shell + tail root, both hip motors
    pv = to_link_frame(pieces["pelvis"], 0.0, org["pelvis"], lift)
    L["pelvis"].mesh_file = pv
    z_cl, z_wl = (hz["crotch"] - hz["hip"]) * MM, (hz["waist"] - hz["hip"]) * MM
    L["pelvis"].items.append(MP.shell_props(pv, drop_planes=(z_wl,)))   # the shorts' underside is real wall
    hs, hc = LY.hip_case()
    for sgn in (1, -1):
        _motor(L["pelvis"], hs, hc + np.array([0, sgn * y_hip_mm, 0]))
    L["pelvis"].items.append(MP.point_props(2 * P.BRACKET_MASS_KG, (0, 0, 0)))
    L["pelvis"].items.append(MP.point_props(P.ELECTRONICS_KG, np.array(P.ELECTRONICS_POS) * MM))
    belly = pv.vertices[np.abs(pv.vertices[:, 1]) < P.ARM_X_NATIVE * S]
    L["pelvis"].collision.append(("box", _fit_box(belly)))

    # ---- torso: waist motor in the chest, battery in the backpack
    tr = to_link_frame(pieces["torso"], 0.0, org["torso"], lift)
    L["torso"].mesh_file = tr
    L["torso"].items.append(MP.shell_props(tr, drop_planes=(0.0,)))
    ws, wc = LY.waist_case()
    _motor(L["torso"], ws, wc)
    L["torso"].items.append(MP.point_props(P.BRACKET_MASS_KG, (0, 0, 0)))
    L["torso"].items.append(MP.point_props(P.HEAD_EXTRA_KG, (0, 0, 0.62 * tr.bounds[1][2])))
    z_neck = (P.NECK_NATIVE * P.MESH_SCALE + lift) * MM - org["torso"][2]
    L["torso"].collision.extend(torso_collision(tr, z_neck, P.ARM_X_NATIVE * S, P.HEAD_BANDS))

    # ---- legs: straight shell with the roll motor standing inside
    ls, lc = LY.leg_motor()
    for side, sgn in (("l", 1), ("r", -1)):
        key = f"leg_{side}"
        c_native = (c_l if sgn > 0 else c_r)[0]
        lg = to_link_frame(pieces[key], c_native - sgn * y_hip_mm / P.MESH_SCALE, org[key], lift)
        L[key].mesh_file = lg
        L[key].items.append(MP.shell_props(lg, drop_planes=((hz["crotch"] - hz["hip"]) * MM,)))
        _motor(L[key], ls, lc + np.array([0, 0, z_r - hz["hip"]]))
        L[key].collision.append(("box", _fit_box(lg.vertices)))

    # ---- ankle bracket: joins the roll horn (leg) to the pitch horn (foot), no motor
    # modelled as a U-bracket block spanning the two shafts (needs a real inertia)
    span = (z_r - z_p) * MM
    for side in ("l", "r"):
        L[f"ankle_{side}"].items.append(
            MP.box_props(2 * P.BRACKET_MASS_KG, (0.020, 0.045, span), (0, 0, -span / 2)))

    # ---- feet: the bun shell with the pitch motor inside
    for side, sgn in (("l", 1), ("r", -1)):
        key = f"foot_{side}"
        fm = LY.foot_mesh(f)
        fm.vertices[:, 1] *= sgn
        if sgn < 0:
            fm.invert()                                   # mirroring flips the winding
        fm.apply_scale(MM)
        L[key].mesh_file = fm
        L[key].items.append(MP.shell_props(fm))
        L[key].items.append(MP.point_props(0.015, (f["cx"] * MM, sgn * f["cy"] * MM, -z_p * MM)))
        ps, pc = fl["pitch"]
        _motor(L[key], ps, pc * np.array([1, sgn, 1]))
        L[key].collision.append(("mesh", key))
        L[key].z_sole_mm = z_p

    joints = [
        ("waist_yaw",   "pelvis", "torso",  (0, 0, 1), org["torso"] - org["pelvis"], "waist_yaw"),
        ("hip_pitch_l", "pelvis", "leg_l",  (0, 1, 0), org["leg_l"] - org["pelvis"], "hip_pitch"),
        ("hip_pitch_r", "pelvis", "leg_r",  (0, 1, 0), org["leg_r"] - org["pelvis"], "hip_pitch"),
        ("ankle_roll_l", "leg_l", "ankle_l", (1, 0, 0), org["ankle_l"] - org["leg_l"], "ankle_roll"),
        ("ankle_roll_r", "leg_r", "ankle_r", (1, 0, 0), org["ankle_r"] - org["leg_r"], "ankle_roll"),
        ("ankle_pitch_l", "ankle_l", "foot_l", (0, 1, 0), org["foot_l"] - org["ankle_l"], "ankle_pitch"),
        ("ankle_pitch_r", "ankle_r", "foot_r", (0, 1, 0), org["foot_r"] - org["ankle_r"], "ankle_pitch"),
    ]
    meta = dict(org=org, y_hip=y_hip_mm * MM, x_axis=x_axis, heights=hz, foot=f,
                z_pitch=z_p, z_roll=z_r, z_leg_bottom=z_leg_bottom, leg_len=leg_len,
                leg_half=leg_half)
    meta["inside"] = motors_inside(L, org)
    return L, joints, meta


def motors_inside(L, org, n=6):
    """Fraction of each motor case inside the character's shells (neutral pose)."""
    shells = [(l.mesh_file.copy(), org[k]) for k, l in L.items() if l.mesh_file is not None]
    world = []
    for m, o in shells:
        m.apply_translation(o)
        world.append(m)
    g = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    out = {}
    for k, l in L.items():
        for i, (size, c) in enumerate(l.motors):
            pts = grid * size + c + org[k]
            inside = np.zeros(len(pts), bool)
            for m in world:
                sel = ~inside
                if not sel.any():
                    break
                lo, hi = m.bounds
                near = sel & np.all((pts >= lo) & (pts <= hi), axis=1)
                if near.any():
                    inside[near] |= m.contains(pts[near])
            out[f"{k}#{i}"] = inside.mean()
    return out


# --------------------------------------------------------------------------- #
#  URDF
# --------------------------------------------------------------------------- #
RGBA = {
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
           '<!-- Haetburi biped - generated by tools/build_model.py, do not edit by hand -->',
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
    L, joints, meta = build()
    for name, link in L.items():
        if link.mesh_file is not None:
            link.mesh_file.export(os.path.join(P.MESH_DIR, f"{name}.stl"))
    urdf = os.path.join(P.OUT_DIR, "haetburi.urdf")
    write_urdf(L, joints, urdf)
    scene = os.path.join(P.OUT_DIR, "haetburi_scene.xml")
    write_mjcf.write(L, joints, meta, scene)
    print("wrote", urdf)
    print("wrote", scene)

    tot = MP.combine([l.props() for l in L.values()])
    print(f"\n{'link':9s} {'mass kg':>8s}  com (m)")
    for n, l in L.items():
        m, c, _ = l.props()
        print(f"{n:9s} {m:8.3f}  {np.round(c + meta['org'][n], 4)}")
    # total com in the robot frame (links at their nominal standing pose)
    items = [(l.props()[0], l.props()[1] + meta["org"][n], l.props()[2]) for n, l in L.items()]
    M, C, _ = MP.combine(items)
    print(f"\ntotal mass {M:.3f} kg   com {np.round(C,4)} m   com height {C[2]*1000:.0f} mm")
    hz, f = meta["heights"], meta["foot"]
    print(f"standing height {hz['top']:.0f} mm (character {P.CHAR_HEIGHT:.0f} mm, lifted {hz['lift']:.0f} mm)")
    print(f"foot {f['L']:.1f} x {f['W']:.1f} x {f['H']:.1f} mm, pitch axis {meta['z_pitch']:.1f} mm above the sole")
    print(f"leg {2*meta['leg_half'][1]:.1f} wide x {2*meta['leg_half'][0]:.1f} deep, "
          f"{meta['leg_len']:.1f} mm from crotch to bottom, roll axis {meta['z_roll']:.1f} mm")
    foot_top = f["H"]
    print(f"ankle gap: leg shell starts {meta['z_leg_bottom'] - foot_top:.1f} mm above the foot top")
    print("motor volume inside the shell:")
    for k, v in meta["inside"].items():
        print(f"  {k:10s} {v*100:5.1f} %")
    return L, joints, meta


if __name__ == "__main__":
    main()
