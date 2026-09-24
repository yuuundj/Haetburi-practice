"""Motor placement and shell sizing for the split ankle (robot frame, mm).

Per leg, x forward, y lateral (layout is symmetric about the leg centre),
z up.  Motor frame: shaft along D, long side H with the shaft 11.25 mm from
one end, short side W.
"""
import numpy as np
import shapely.geometry as sg
import trimesh

import ankle_search as AS
import haetburi_params as P

S = P.MESH_SCALE
DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
        "+z": (0, 0, 1), "-z": (0, 0, -1)}
H_MID = (P.MOTOR_H - 2 * P.MOTOR_AXIS_FROM_END) / 2.0     # case centre along H, from the shaft
HIP_ABOVE_CROTCH = 1.0 + P.MOTOR_AXIS_FROM_END           # hip case sits 1 mm above the crotch


def _aabb(shaft, h_dir, shift=0.0):
    """Axis-aligned size and centre of a motor case, shaft through the origin."""
    d = AS.AX[shaft]
    h = np.array(DIRS[h_dir], float)
    w = np.cross(d, h)
    R = np.stack([d, h, w], axis=1)
    size = np.abs(R) @ np.array([P.MOTOR_D, P.MOTOR_H, P.MOTOR_W])
    centre = R @ np.array([shift, H_MID, 0.0])
    return size, centre


def _rot(axis, deg):
    return AS.rot(axis, deg)


def _corners(size, centre, inflate=0.0):
    s = np.array(np.meshgrid(*[(-1, 1)] * 3)).reshape(3, -1).T
    return centre + s * (np.asarray(size) / 2.0 + inflate)


# ------------------------------------------------------------------- the foot
def foot_layout():
    """Pitch motor and the bun around it, foot frame (origin on the pitch axis)."""
    size, c = _aabb("y", P.PITCH_H_DIR)
    horn = size + np.array([0, P.MOTOR_HORN, 0])               # horn on one y face
    z0 = c[2] - size[2] / 2 - P.FLOOR
    reach = np.array([[-P.FOOT_HEEL_REACH, 0, z0 + 0.5], [P.FOOT_TOE_REACH, 0, z0 + 0.5]])
    env = np.vstack([_corners(horn, c, inflate=P.WALL), reach])
    f = AS.fit_bun(env, z0, (P.FOOT_PLAN_EXPONENT, P.FOOT_SIDE_EXPONENT))
    if f is None:
        raise RuntimeError("no foot bun fits between the legs")
    f.update(z0=z0)
    return dict(z_pitch=-z0, foot=f, pitch=(size, c))


def _bun_value(f, pts, grow=0.0):
    a, b, c = f["a"] + grow, f["b"] + grow, f["c"] + grow
    p, q = f["p"], f["q"]
    return ((np.abs(pts[:, 0] - f["cx"]) / a) ** p
            + (np.abs(pts[:, 1] - f["cy"]) / b) ** p) ** (q / p) \
        + (np.abs(pts[:, 2] - f["cz"]) / c) ** q


def foot_inside(f, pts, grow=0.0):
    return _bun_value(f, pts, grow) < 1.0


def foot_mesh(f):
    """Watertight foot solid in the foot frame (mm)."""
    sph = trimesh.creation.icosphere(subdivisions=5)
    v = sph.vertices
    unit = dict(f, cx=0.0, cy=0.0, cz=0.0)
    r = _bun_value(unit, v) ** (-1.0 / f["q"])                   # the value is degree-q homogeneous
    sph.vertices = v * r[:, None] + np.array([f["cx"], f["cy"], f["cz"]])
    return sph.slice_plane([0, 0, f["z0"]], [0, 0, 1], cap=True)


# -------------------------------------------------------------------- the leg
def leg_motor():
    """Roll motor in the leg, relative to the roll axis: standing, long end up."""
    return _aabb("x", "+z")


def leg_section():
    """Rounded-square leg section (outer half sizes, mm) that holds the roll
    motor plus its horn with a full wall all round."""
    size, _ = leg_motor()
    hx = (size[0] + P.MOTOR_HORN) / 2.0                        # fore-aft, horn on one x face
    hy = size[1] / 2.0
    k = 2.0 ** (1.0 / P.LEG_EXPONENT)                           # corner of a box in a superellipse
    return hx * k + P.WALL, hy * k + P.WALL                     # (half depth, half width)


def leg_polygon(half_x, half_y, n, pts=128):
    """Superellipse outline in the (x_world, y_world) plane."""
    t = np.linspace(0, 2 * np.pi, pts, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.c_[half_x * np.sign(c) * np.abs(c) ** (2.0 / n),
                 half_y * np.sign(s) * np.abs(s) ** (2.0 / n)]


def hip_clear_depth():
    """How far below the hip axis the top of the leg motor must stay so that the
    hip can swing through its whole range without the two cases touching."""
    hs, hc = hip_case()
    hip = (hc, np.eye(3), hs / 2.0 + np.array([0, P.MOTOR_HORN / 2, 0]))
    ls, _ = leg_motor()
    half = ls / 2.0 + np.array([P.MOTOR_HORN / 2, 0, 0])
    lo, hi = P.LIM["hip_pitch"]
    for d in np.arange(0.0, 80.0, 0.25):
        c = np.array([0, 0, -d - half[2]])
        if not any(AS.obb_overlap(hip, (_rot("y", a) @ c, _rot("y", a), half), P.CLEAR)
                   for a in np.linspace(lo, hi, 61)):
            return d
    raise RuntimeError("leg motor cannot clear the hip motor")


def leg_length():
    """Crotch to leg bottom (mm) and roll-axis height above the leg bottom."""
    ls, lc = leg_motor()
    motor_bottom_below_shaft = ls[2] / 2 - lc[2]                 # 11.25
    d = hip_clear_depth()
    length = d + ls[2] + P.WALL - HIP_ABOVE_CROTCH
    roll_above_bottom = P.WALL + motor_bottom_below_shaft
    return length, roll_above_bottom, d


def leg_clear_height(f, leg_rel, roll_above_bottom, z_start=0.0):
    """Lowest leg-bottom height above the pitch axis at which the leg shell
    never touches its foot anywhere in the ankle range.
    leg_rel: leg shell points relative to (leg centre, leg bottom), mm."""
    plo, phi = P.LIM["ankle_pitch"]
    rlo, rhi = P.LIM["ankle_roll"]
    z = z_start
    while True:
        q0 = leg_rel + np.array([0, 0, z])
        r0 = np.array([0, 0, z + roll_above_bottom])
        hit = False
        for rr in np.linspace(rlo, rhi, 7):
            q = (q0 - r0) @ _rot("x", rr).T + r0
            for pp in np.linspace(plo, phi, 9):
                if foot_inside(f, q @ _rot("y", pp).T, grow=P.LEG_FOOT_CLEAR).any():
                    hit = True
                    break
            if hit:
                break
        if not hit:
            return z
        z += 0.5


# ------------------------------------------------------------------- the body
def body_heights(z_leg_bottom, leg_len):
    """World heights once the character is lifted onto legs of length leg_len."""
    z_crotch = z_leg_bottom + leg_len
    lift = z_crotch - P.CUT_LEG_TOP * S
    return dict(lift=lift, crotch=z_crotch, waist=P.CUT_WAIST * S + lift,
                hip=z_crotch + HIP_ABOVE_CROTCH, top=P.SRC_HEIGHT * S + lift)


def hip_case():
    """Hip motor in the pelvis, relative to the hip axis: standing, horn down."""
    return _aabb("y", "+z")


def waist_case():
    """Waist motor in the chest, relative to the waist axis: shaft vertical,
    horn down onto the pelvis, long end toward the backpack."""
    d = np.array([0, 0, 1.0]); h = np.array([-1.0, 0, 0]); w = np.cross(d, h)
    R = np.stack([d, h, w], axis=1)
    size = np.abs(R) @ np.array([P.MOTOR_D, P.MOTOR_H, P.MOTOR_W])
    centre = R @ np.array([P.MOTOR_D / 2.0 + 0.5, H_MID, 0.0])
    return size, centre
