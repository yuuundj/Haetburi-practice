"""Geometry helpers for the robot built on the new base character
(modelings/haetburi_display_parts.stl via tools/make_display_model.py).

The STL is open surfaces with holes: about 33 500 loose triangles patch the body,
feet, arms and backpack.  Cutting and drawing use the surfaces as they are; any
"is this inside the character" question (motor fit, containment) uses voxel
solids made here: every z slice of a part's surface (loose triangles included) is
closed, split into its loops, and each loop's convex hull is filled.  The parts
are convex-ish in plan (belly, legs, feet, backpack), so the hull is a fair, and
slightly generous, stand-in (arcs of one loop are grouped first) for the inside.

All coordinates are native STL units (120 tall, -y = face, z up from the sole).
"""
import numpy as np
import trimesh
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage.morphology import convex_hull_image

import make_display_model as MD

PITCH = 0.1                 # voxel size, native units (0.64 mm at the robot's scale)


class Solid:
    def __init__(self, grid, lo, pitch):
        self.grid, self.lo, self.pitch = grid, np.asarray(lo, float), pitch

    def inside(self, pts):
        idx = np.floor((np.asarray(pts) - self.lo) / self.pitch).astype(int)
        ok = np.all((idx >= 0) & (idx < self.grid.shape), axis=1)
        out = np.zeros(len(idx), bool)
        out[ok] = self.grid[tuple(idx[ok].T)]
        return out

    @property
    def top(self):
        return self.lo[2] + (np.argwhere(self.grid.any(axis=(0, 1)))[-1][0] + 1) * self.pitch


def part_surfaces():
    """The 8 STL parts with their loose triangles (assigned to the nearest part)."""
    mesh = trimesh.load(MD.NEW)
    comps = sorted(mesh.split(only_watertight=False), key=lambda c: -c.area)
    big, loose = comps[:8], comps[8:]
    verts = np.vstack([b.vertices for b in big])
    owner = np.concatenate([np.full(len(b.vertices), i) for i, b in enumerate(big)])
    _, near = cKDTree(verts).query(np.array([c.centroid for c in loose]))
    out = {}
    for i, name in enumerate(MD.PART_NAMES):
        mine = [loose[j] for j in np.where(owner[near] == i)[0]]
        out[name] = trimesh.util.concatenate([big[i]] + mine)
    return out


def solid(surface, pitch=PITCH, close=2, group=6):
    pts, _ = trimesh.sample.sample_surface(surface, int(surface.area / (pitch * 0.4) ** 2), seed=1)
    pts = np.vstack([pts, surface.vertices])
    lo = surface.bounds[0] - pitch * (close + 2)
    idx = np.floor((pts - lo) / pitch).astype(int)
    shell = np.zeros(idx.max(axis=0) + close + 3, bool)
    shell[tuple(idx.T)] = True
    grid = np.zeros_like(shell)
    for k in range(shell.shape[2]):
        sl = ndimage.binary_closing(shell[:, :, k], iterations=close)
        if sl.sum() < 4:
            continue
        # a loop's surface samples can break into arcs: arcs closer than 2 * group
        # voxels are one loop (the two legs are ~19 voxels apart)
        lab, n = ndimage.label(ndimage.binary_dilation(sl, iterations=group))
        for j in range(1, n + 1):
            comp = sl & (lab == j)
            if comp.sum() > 3:
                grid[:, :, k] |= convex_hull_image(comp)
    return Solid(grid, lo, pitch)


# ------------------------------------------------------------ motor in the foot
def motor_frame(theta_deg):
    """Shaft along y_stl (the robot's fore-aft roll axis); the case's long side H
    turned theta from +z toward +x_stl; W completes the frame."""
    t = np.radians(theta_deg)
    s = np.array([0.0, 1.0, 0.0])
    h = np.array([np.sin(t), 0.0, np.cos(t)])
    return s, h, np.cross(s, h)


def case_extent(P, walls=True, horn_front=True):
    """(lo, hi) of the case along (s, h, w) in mm, measured from the shaft;
    with walls: printed wall all round, horn on one end of the shaft."""
    wall = P.WALL if walls else 0.0
    horn = P.MOTOR_HORN if walls else 0.0
    lo = np.array([-P.MOTOR_D / 2 - wall - (horn if horn_front else 0.0),
                   -P.MOTOR_AXIS_FROM_END - wall, -P.MOTOR_W / 2 - wall])
    hi = np.array([P.MOTOR_D / 2 + wall + (0.0 if horn_front else horn),
                   P.MOTOR_H - P.MOTOR_AXIS_FROM_END + wall, P.MOTOR_W / 2 + wall])
    return lo, hi


def fit_motor(P, foot, k, theta_deg):
    """Shaft positions (native) at which the walled case fits inside the foot solid
    at k mm per native unit with the case FLOOR above the sole; the lowest one.
    None if it does not fit."""
    s, h, w = motor_frame(theta_deg)
    R = np.stack([s, h, w])                               # rows: local axes in native xyz
    lo, hi = case_extent(P)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    world = corners @ R / k
    bmin, bmax = world.min(axis=0), world.max(axis=0)
    p = foot.pitch
    n = np.ceil((bmax - bmin) / p).astype(int) + 1
    axes = [bmin[i] + np.arange(n[i]) * p for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1) * k
    loc = g @ R.T
    mask = np.all((loc >= lo) & (loc <= hi), axis=-1).astype(np.float32)
    if any(mask.shape[i] > foot.grid.shape[i] for i in range(3)):
        return None
    hits = signal.fftconvolve(foot.grid.astype(np.float32), mask[::-1, ::-1, ::-1], mode="valid")
    ok = np.argwhere(hits >= mask.sum() - 0.5)
    if not len(ok):
        return None
    shaft = foot.lo + ok * p - bmin
    clo, chi = case_extent(P, walls=False)
    ccorn = np.array([[a, b, c] for a in (clo[0], chi[0]) for b in (clo[1], chi[1]) for c in (clo[2], chi[2])])
    case_bottom = shaft[:, 2] * k + (ccorn @ R)[:, 2].min()
    shaft = shaft[case_bottom >= P.FLOOR]
    if not len(shaft):
        return None
    # lowest axis; among those, the one nearest the middle of the feasible set
    low = shaft[shaft[:, 2] <= shaft[:, 2].min() + 1e-6]
    return low[np.argmin(np.linalg.norm(low - low.mean(axis=0), axis=1))]


def min_scale(P, foot, theta_deg, k_lo=4.0, k_hi=9.0, tol=0.01):
    if fit_motor(P, foot, k_hi, theta_deg) is None:
        return None
    while k_hi - k_lo > tol:
        k = (k_lo + k_hi) / 2
        if fit_motor(P, foot, k, theta_deg) is None:
            k_lo = k
        else:
            k_hi = k
    return k_hi


def leg_lift(P, leg_pts, foot, shaft, k, step=0.05, clear_mm=None):
    """How far (native) the leg and everything above it must rise so the leg
    never comes within LEG_FOOT_CLEAR of its foot over the ankle roll range.
    The roll axis runs fore-aft (along y_stl) through `shaft`."""
    clear_mm = P.LEG_FOOT_CLEAR if clear_mm is None else clear_mm
    grow = int(np.ceil(clear_mm / k / foot.pitch))
    padded = np.pad(foot.grid, grow + 1)                  # room to grow past the grid's edge
    grown = Solid(ndimage.binary_dilation(padded, iterations=grow),
                  foot.lo - (grow + 1) * foot.pitch, foot.pitch)
    lo, hi = P.LIM["ankle_roll"]
    rots = []
    for a in np.radians(np.linspace(lo, hi, 13)):
        c, s = np.cos(a), np.sin(a)
        rots.append(np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]))
    for dz in np.arange(0.0, 10.0, step):
        q = leg_pts + np.array([0.0, 0.0, dz]) - shaft
        if not any(grown.inside(q @ R.T + shaft).any() for R in rots):
            return dz
    raise RuntimeError("leg cannot clear its foot")


def motor_fraction_inside(solids, size_native, centre_native, n=6):
    """Share of a motor case (axis-aligned box, native units) inside any solid."""
    g = (np.arange(n) + 0.5) / n - 0.5
    pts = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T * size_native + centre_native
    inside = np.zeros(len(pts), bool)
    for s in solids:
        inside |= s.inside(pts)
    return float(inside.mean())


def flat_sole(foot, zc, fillet, n=180):
    """Give the foot a flat sole as big as its own outline at height zc: below zc
    the bun is replaced by straight sides down to the floor, rounded into the sole
    with a fillet of radius `fillet`.  Seen from above the foot is unchanged."""
    import shapely.geometry as sg
    upper = foot.slice_plane([0, 0, zc], [0, 0, 1])
    sec = foot.section(plane_origin=[0, 0, zc], plane_normal=[0, 0, 1])
    loop = max(sec.discrete, key=lambda l: sg.Polygon(l[:, :2]).area)
    poly = sg.Polygon(loop[:, :2]).buffer(0)
    c = np.array(poly.centroid.coords[0])
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    far = 10 * max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    radius = np.empty(n)
    for i, a in enumerate(phi):
        ray = sg.LineString([c, c + far * np.array([np.cos(a), np.sin(a)])])
        hit = ray.intersection(poly.exterior)
        pts = np.array([g.coords[0] for g in getattr(hit, "geoms", [hit])])
        radius[i] = np.linalg.norm(pts - c, axis=1).max()
    t = np.linspace(0, np.pi / 2, 7)
    levels = [(fillet * (1 - np.cos(a)), fillet * (1 - np.sin(a))) for a in t]   # (z, inset)
    levels.append((zc, 0.0))
    rings = []
    for z, inset in levels:
        r = radius - inset
        rings.append(np.c_[c[0] + r * np.cos(phi), c[1] + r * np.sin(phi), np.full(n, z)])
    verts = np.vstack(rings + [[c[0], c[1], 0.0]])
    faces = []
    for j in range(len(rings) - 1):
        a0, b0 = j * n, (j + 1) * n
        for i in range(n):
            i2 = (i + 1) % n
            faces += [[a0 + i, a0 + i2, b0 + i2], [a0 + i, b0 + i2, b0 + i]]
    centre = len(verts) - 1
    faces += [[centre, i2, i] for i, i2 in ((i, (i + 1) % n) for i in range(n))]
    lower = trimesh.Trimesh(verts, np.array(faces), process=False)
    lower.fix_normals()
    return trimesh.util.concatenate([upper, lower])


def union_solid(bounds, tests, pitch=PITCH):
    """Voxel solid of everything inside any of `tests` (callables pts -> bool)
    within bounds ((lo), (hi)), native units."""
    lo, hi = np.asarray(bounds[0], float) - pitch, np.asarray(bounds[1], float) + pitch
    n = np.ceil((hi - lo) / pitch).astype(int)
    axes = [lo[i] + (np.arange(n[i]) + 0.5) * pitch for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(pts), bool)
    for t in tests:
        inside |= t(pts)
    return Solid(inside.reshape(n), lo, pitch)


def in_box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return lambda pts: np.all((pts >= lo) & (pts <= hi), axis=1)


def outline(mesh, z):
    """Largest closed loop of a z section, as a shapely polygon (native xy); the
    convex hull of the section's points where holes keep the loop from closing."""
    import shapely.geometry as sg
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    loops = [l for l in sec.discrete if len(l) > 3 and np.allclose(l[0], l[-1])]
    if loops:
        return max((sg.Polygon(l[:, :2]).buffer(0) for l in loops), key=lambda q: q.area)
    return sg.MultiPoint(sec.vertices[:, :2]).convex_hull


def cap(poly, z, up):
    """Flat triangulated polygon at height z, facing up or down."""
    v2, f = trimesh.creation.triangulate_polygon(poly)
    m = trimesh.Trimesh(np.c_[v2, np.full(len(v2), z)], f, process=False)
    if (m.face_normals[:, 2].mean() > 0) != up:
        m.invert()
    return m


# ------------------------------------------------- two ankle motors in one foot
def rotation(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _placements(P, grid_solid, grid, R, k):
    """All shaft positions (native) where the walled case with local frame R
    (rows: shaft, long side, short side) fits inside `grid`, case FLOOR above the sole."""
    p = grid_solid.pitch
    lo, hi = case_extent(P)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    w = corners @ R / k
    bmin, bmax = w.min(axis=0), w.max(axis=0)
    n = np.ceil((bmax - bmin) / p).astype(int) + 1
    axes = [bmin[i] + np.arange(n[i]) * p for i in range(3)]
    loc = np.stack(np.meshgrid(*axes, indexing="ij"), -1) * k @ R.T
    mask = np.all((loc >= lo) & (loc <= hi), axis=-1).astype(np.float32)
    if any(mask.shape[i] > grid.shape[i] for i in range(3)):
        return np.zeros((0, 3))
    hits = signal.fftconvolve(grid.astype(np.float32), mask[::-1, ::-1, ::-1], mode="valid")
    shaft = grid_solid.lo + np.argwhere(hits >= mask.sum() - 0.5) * p - bmin
    clo, chi = case_extent(P, walls=False)
    cc = np.array([[a, b, c] for a in (clo[0], chi[0]) for b in (clo[1], chi[1]) for c in (clo[2], chi[2])])
    return shaft[shaft[:, 2] * k + (cc @ R)[:, 2].min() >= P.FLOOR]


def in_case(P, R, shaft, k, grow_mm=0.0):
    """Point test for a bare case (frame R, shaft at `shaft`, native), grown by grow_mm."""
    lo, hi = case_extent(P, walls=False)
    return lambda pts: np.all(((pts - shaft) @ R.T * k >= lo - grow_mm)
                              & ((pts - shaft) @ R.T * k <= hi + grow_mm), axis=1)


def fit_two(P, foot, k, RA, RB, axis_a, range_a, n_try=40, seed=0, b_target_y=None, a_hint=None,
            a_target_x=None):
    """Motor A fixed in the foot, motor B on the link A turns (B sweeps +-range_a
    about A's shaft).  Returns (shaft A, shaft B), or None.  B is the lowest one;
    with b_target_y, the one whose shaft is nearest that fore-aft position (the
    ankle pitch axis belongs under the robot's centre of mass), over all tries.
    With a_target_x, A's shaft is also pulled toward that lateral position (the
    ankle roll axis belongs near the inner edge: the stance roll motor has to push
    the whole robot over onto that foot); the A placements nearest it are tried."""
    cand = _placements(P, foot, foot.grid, RA, k)
    if not len(cand):
        return None
    if a_target_x is not None:
        cand = cand[np.argsort(np.abs(cand[:, 0] - a_target_x) + 0.01 * np.abs(cand[:, 1]))[:n_try]]
    elif a_hint is None:
        rng = np.random.default_rng(seed)
        cand = cand[rng.choice(len(cand), min(n_try, len(cand)), replace=False)]
    else:                                  # try the A placements nearest a hint first
        cand = cand[np.argsort(np.linalg.norm(cand - a_hint, axis=1))[:n_try]]
    best = None
    pts = np.stack(np.meshgrid(*[foot.lo[i] + (np.arange(foot.grid.shape[i]) + 0.5) * foot.pitch
                                 for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    for sA in cand:
        free = foot.grid.reshape(-1) & ~in_case(P, RA, sA, k, P.CLEAR)(pts)
        fr = Solid(free.reshape(foot.grid.shape), foot.lo, foot.pitch)
        ok = np.ones(len(pts), bool)
        for a in np.linspace(-range_a, range_a, 7):
            ok &= fr.inside((pts - sA) @ rotation(axis_a, a).T + sA)
        sB = _placements(P, foot, ok.reshape(foot.grid.shape), RB, k)
        if not len(sB):
            continue
        if b_target_y is None:
            return sA, sB[np.argmin(sB[:, 2])]
        j = np.argmin(np.abs(sB[:, 1] - b_target_y) + 0.01 * sB[:, 2])
        score = abs(sB[j, 1] - b_target_y) + (0.0 if a_target_x is None else abs(sA[0] - a_target_x))
        if best is None or score < best[2]:
            best = (sA, sB[j], score)
    if best is not None:
        return best[0], best[1]
    return best


def leg_lift_two(P, leg_pts, foot_obst, ankle_obst, sA, axis_a, range_a, sB, axis_b, range_b, k,
                 step=0.05, clear_mm=None):
    """Lift (native) at which the leg clears the foot side (foot + its motor) over
    both joints and the middle link's motor over the upper joint.  The leg turns
    about B (upper joint); B rides on the link that turns about A."""
    clear_mm = P.LEG_FOOT_CLEAR if clear_mm is None else clear_mm
    grow = int(np.ceil(clear_mm / k / foot_obst.pitch))
    def grown(s):
        g = np.pad(s.grid, grow + 1)
        return Solid(ndimage.binary_dilation(g, iterations=grow), s.lo - (grow + 1) * s.pitch, s.pitch)
    gf, ga = grown(foot_obst), grown(ankle_obst)
    RA = [rotation(axis_a, a) for a in np.linspace(-range_a, range_a, 7)]
    RB = [rotation(axis_b, b) for b in np.linspace(-range_b, range_b, 7)]
    for dz in np.arange(0.0, 15.0, step):
        q = leg_pts + np.array([0.0, 0.0, dz])
        hit = False
        for rb in RB:
            qb = (q - sB) @ rb.T + sB                       # leg in the middle link's frame
            if ga.inside(qb).any():
                hit = True
                break
            for ra in RA:
                if gf.inside((qb - sA) @ ra.T + sA).any():
                    hit = True
                    break
            if hit:
                break
        if not hit:
            return dz
    raise RuntimeError("leg cannot clear its foot")


def case_box_robot(P, R, k):
    """Axis-aligned case (no horn) in the robot frame, mm, relative to its shaft:
    (size, centre).  Native (x, y, z) maps to robot (-y, x, z)."""
    lo, hi = case_extent(P, walls=False)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]) @ R
    rob = np.c_[-corners[:, 1], corners[:, 0], corners[:, 2]]
    return rob.max(axis=0) - rob.min(axis=0), (rob.max(axis=0) + rob.min(axis=0)) / 2


def two_fit_ok(P, foot, k, RA, RB, sA, sB, axis_a, range_a, step=0.1):
    """Check a given two-motor placement: walled A inside the foot, walled B inside
    the foot and clear of A over A's whole range."""
    def surface(R, s):
        lo, hi = case_extent(P)
        g = [np.arange(lo[i], hi[i] + 1e-9, step * k) for i in range(3)]
        pts = []
        for i in range(3):
            for v in (lo[i], hi[i]):
                o = [g[j] for j in range(3) if j != i]
                a, b = np.meshgrid(*o, indexing="ij")
                c = [None] * 3
                c[i] = np.full(a.size, v)
                jj = [j for j in range(3) if j != i]
                c[jj[0]], c[jj[1]] = a.ravel(), b.ravel()
                pts.append(np.stack(c, 1))
        return np.vstack(pts) @ R / k + s
    if not foot.inside(surface(RA, sA)).all():
        return False
    inA = in_case(P, RA, sA, k, P.CLEAR)
    pb = surface(RB, sB)
    for a in np.linspace(-range_a, range_a, 7):
        q = (pb - sA) @ rotation(axis_a, a).T + sA
        if not foot.inside(q).all() or inA(q).any():
            return False
    return True


# ------------------------------------------- stacked ankle: roll in the leg, pitch in the foot
def leg_lift_stacked(P, leg_pts, up_pt, foot_obst, s_low, k, step=0.05, clear_mm=None,
                     up="roll"):
    """Lift (native) at which the lengthened leg clears the foot side (foot + its
    motor): the leg turns about the upper ankle axis (through up_pt, which rises
    with the leg), and with it about the lower one (through s_low, in the foot).
    up = "roll": upper axis fore-aft, lower lateral; "pitch": the other way round."""
    clear_mm = P.LEG_FOOT_CLEAR if clear_mm is None else clear_mm
    grow = int(np.ceil(clear_mm / k / foot_obst.pitch))
    g = np.pad(foot_obst.grid, grow + 1)
    gf = Solid(ndimage.binary_dilation(g, iterations=grow), foot_obst.lo - (grow + 1) * foot_obst.pitch,
               foot_obst.pitch)
    rr, rp = P.LIM["ankle_roll"][1], P.LIM["ankle_pitch"][1]
    R_roll = [rotation((0, 1, 0), b) for b in np.linspace(-rr, rr, 7)]
    R_pitch = [rotation((1, 0, 0), a) for a in np.linspace(-rp, rp, 7)]
    R_up, R_low = (R_roll, R_pitch) if up == "roll" else (R_pitch, R_roll)
    for dz in np.arange(0.0, 15.0, step):
        q = leg_pts + np.array([0.0, 0.0, dz])
        r = np.asarray(up_pt) + np.array([0.0, 0.0, dz])
        if not any(gf.inside(((q - r) @ ru.T + r - s_low) @ rl.T + s_low).any()
                   for ru in R_up for rl in R_low):
            return dz
    raise RuntimeError("leg cannot clear its foot")


def liner(solid, keep=None, erode=2, faces=20000):
    """A closed lining just inside a part (the voxel solid shrunk by `erode` voxels),
    drawn in the shell colour so holes in the STL surface show shell, not the
    motors behind them.  keep(pts) -> bool limits it to one link's share.
    Visual only: it carries no mass."""
    from skimage import measure
    g = solid.grid.copy()
    if keep is not None:
        pts = np.stack(np.meshgrid(*[solid.lo[i] + (np.arange(g.shape[i]) + 0.5) * solid.pitch
                                     for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        g &= keep(pts).reshape(g.shape)
    g = ndimage.binary_erosion(g, iterations=erode)
    if not g.any():
        return None
    v, f, _, _ = measure.marching_cubes(np.pad(g, 1).astype(np.float32), 0.5)
    m = trimesh.Trimesh((v - 1 + 0.5) * solid.pitch + solid.lo, f[:, ::-1], process=True)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m


def case_surface(P, step_mm=1.0, walls=True):
    """Points on the six faces of the (walled) case, local (s, h, w) mm from the shaft."""
    lo, hi = case_extent(P, walls)
    g = [np.arange(lo[i], hi[i] + 1e-9, step_mm) for i in range(3)]
    pts = []
    for i in range(3):
        o = [j for j in range(3) if j != i]
        a, b = np.meshgrid(g[o[0]], g[o[1]], indexing="ij")
        for v in (lo[i], hi[i]):
            p = np.zeros((a.size, 3))
            p[:, i], p[:, o[0]], p[:, o[1]] = v, a.ravel(), b.ravel()
            pts.append(p)
    return np.vstack(pts)


def fit_hip_pitch(P, body, k, leg, z_crotch, sgn, ahead_mm, yaw_deg, b_range=8.0, dz_max=6.0,
                  yaw_at_shaft=False):
    """Hip pitch motor under the hip yaw motor: standing (shaft lateral, long side
    up, horn outboard), its walled case on the crotch, turned by the hip yaw about
    the vertical through the leg centre.  The walled case must stay inside the body
    and on its own side of the centre plane over the whole yaw range.  Of the
    placements that do, the one whose shaft is nearest `ahead_mm` in front of the
    leg centre (the upper body's CoM: the hips then hold no standing torque), then
    nearest the leg centre sideways, then lowest.  Returns (shaft, R, ahead, out)
    in native units / mm, or None.  yaw_at_shaft: the yaw axis passes through the
    pitch shaft (it moves forward with it), so the two axes meet (spherical seam)."""
    cx, cy = leg[0], leg[1]
    R = np.array([[-sgn, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])         # rows: s, h, w in native xyz
    off = case_surface(P) @ R / k
    lo, _ = case_extent(P)
    z0 = z_crotch - lo[1] / k                                       # walled bottom on the crotch
    angs = np.radians(np.linspace(-yaw_deg, yaw_deg, 7))
    a_lo, a_hi = min(0.0, ahead_mm), max(0.0, ahead_mm)
    best = None
    for dz in np.arange(0.0, dz_max + 1e-9, 1.0):
        for a in np.arange(a_lo, a_hi + 1e-9, 0.5):
            for b in np.arange(-b_range, b_range + 1e-9, 0.5):
                cost = abs(a - ahead_mm) + 0.05 * abs(b) + 0.3 * dz
                if best is not None and cost >= best[0]:
                    continue
                shaft = np.array([cx + sgn * b / k, cy - a / k, z0 + dz / k])
                yc = shaft[1] if yaw_at_shaft else cy
                rel = shaft + off - [cx, yc, 0]
                ok = True
                for t in angs:
                    c, s = np.cos(t), np.sin(t)
                    q = np.c_[cx + c * rel[:, 0] - s * rel[:, 1], yc + s * rel[:, 0] + c * rel[:, 1], rel[:, 2]]
                    if (sgn * q[:, 0]).min() < 0.3 / k or not body.inside(q).all():
                        ok = False
                        break
                if ok:
                    best = (cost, shaft, a, b)
    if best is None:
        return None
    return best[1], R, best[2], best[3]


def slice_field(mesh, fn):
    """The part of a mesh where fn(points) >= 0; triangles crossing fn = 0 are cut
    along it (linear interpolation of fn on the edges)."""
    V, F = mesh.vertices, mesh.faces
    f = fn(V)
    pos = f[F] >= 0
    n = pos.sum(1)
    tris = [V[F[n == 3]]]

    def cross(a, b):
        fa, fb = f[a][:, None], f[b][:, None]
        return V[a] + fa / (fa - fb) * (V[b] - V[a])

    for cnt, odd in ((1, True), (2, False)):
        Fs, ps = F[n == cnt], pos[n == cnt]
        if not len(Fs):
            continue
        i = np.argmax(ps, 1) if odd else np.argmin(ps, 1)         # the vertex on its own side
        r = np.arange(len(Fs))
        a, b, c = Fs[r, i], Fs[r, (i + 1) % 3], Fs[r, (i + 2) % 3]
        ab, ac = cross(a, b), cross(a, c)
        if odd:                                                  # a kept: one triangle
            tris.append(np.stack([V[a], ab, ac], 1))
        else:                                                    # a dropped: the quad ab-b-c-ac
            tris.append(np.stack([ab, V[b], V[c]], 1))
            tris.append(np.stack([ab, V[c], ac], 1))
    t = np.concatenate(tris)
    out = trimesh.Trimesh(t.reshape(-1, 3), np.arange(len(t) * 3).reshape(-1, 3), process=False)
    out.merge_vertices()
    return out


def _grid_points(solid, sl=None):
    sl = sl or tuple(slice(0, n) for n in solid.grid.shape)
    ax = [solid.lo[i] + (np.arange(solid.grid.shape[i])[sl[i]] + 0.5) * solid.pitch for i in range(3)]
    return np.meshgrid(*ax, indexing="ij")


def hip_seam(P, body_mesh, body, centres, k, gap_mm, target_deg, r0=6.0, r_step=0.1):
    """Split the body between the belly and the two legs.  Outside, along spheres
    centred where each hip's yaw and pitch axes meet: the leg slides along its sphere
    under either joint and never digs into the belly.  Inside, between the legs (the
    belly there is further from the hip than any usable sphere), along the STL's own
    crotch plane, which hip yaw also leaves alone.  The leg is what lies outside its
    sphere (radius R + gap/2) and below the plane; the belly keeps what is inside the
    sphere or above the plane, less the gap.  R is the smallest radius (the seam as
    high as possible) at which the leg pitches +-target_deg both ways, checked on the
    voxel solids, without touching the belly (hip motors included).
    Returns dict(R, legs={sd: mesh}, upper=mesh, leg_solid={sd: Solid}, belly_solid)."""
    g = gap_mm / k
    zc = P.CUT_LEG_TOP
    k1 = int((zc + 1.0 - body.lo[2]) / body.pitch) + 1
    sl = (slice(None), slice(None), slice(0, k1))
    X, Y, Z = _grid_points(body, sl)
    sub = body.grid[sl]
    D = {sd: np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) for sd, c in centres.items()}
    side = {"l": X > 0, "r": X < 0}

    def masks(R):
        legs, belly = {}, body.grid.copy()
        bsub = sub.copy()
        for sd in centres:
            gl = np.zeros_like(body.grid)
            gl[sl] = sub & side[sd] & (D[sd] > R + g / 2) & (Z < zc - g / 2)
            legs[sd] = Solid(gl, body.lo, body.pitch)
            bsub &= ~(side[sd] & (D[sd] > R - g / 2) & (Z < zc + g / 2))
        belly[sl] = bsub
        return legs, Solid(belly, body.lo, body.pitch)

    R = r0
    while True:
        legs, belly = masks(R)
        mv = boundary_points(legs["l"])
        ok = all(not belly.inside(_rot_pitch(mv, centres["l"], sgn * a)).sum() > 3
                 for a in range(2, int(target_deg) + 1, 2) for sgn in (-1, 1))
        if ok:
            break
        R += r_step
        if R > 15:
            raise RuntimeError("no hip sphere gives the leg its pitch range")
    out = {"R": R, "leg_solid": legs, "belly_solid": belly, "legs": {}}
    upper = []
    for sd, c in centres.items():
        sgn = 1 if sd == "l" else -1
        half = body_mesh.slice_plane([0, 0, 0], [sgn, 0, 0])
        d = lambda v, c=c: np.linalg.norm(v - c, axis=1)
        out["legs"][sd] = slice_field(half, lambda v: np.minimum(d(v) - (R + g / 2), (zc - g / 2) - v[:, 2]))
        upper.append(slice_field(half, lambda v: np.maximum((R - g / 2) - d(v), v[:, 2] - (zc + g / 2))))
    out["upper"] = trimesh.util.concatenate(upper)
    return out


def boundary_points(solid, step=1):
    """Centres of a solid's surface voxels (every `step`-th)."""
    g = solid.grid & ~ndimage.binary_erosion(solid.grid)
    idx = np.argwhere(g)[::step]
    return solid.lo + (idx + 0.5) * solid.pitch


def _rot_pitch(pts, c, deg):
    """Robot hip pitch by deg (about robot +y = native +x) about centre c; + swings the leg back."""
    t = np.radians(deg)
    r = pts - c
    return c + np.c_[r[:, 0], r[:, 1] * np.cos(t) - r[:, 2] * np.sin(t), r[:, 1] * np.sin(t) + r[:, 2] * np.cos(t)]


def _rot_yaw(pts, c, deg):
    """Robot hip yaw by deg (about robot +z) about the vertical through c; + turns the toes to robot left."""
    t = np.radians(deg)
    r = pts - c
    return c + np.c_[r[:, 0] * np.cos(t) - r[:, 1] * np.sin(t), r[:, 0] * np.sin(t) + r[:, 1] * np.cos(t), r[:, 2]]


def hip_collisions(moving, obstacles, c, yaw, pitch):
    """{obstacle: number of moving points inside it} with the leg turned by hip yaw, then pitch."""
    q = _rot_yaw(_rot_pitch(moving, c, pitch), c, yaw)
    return {n: int(fn(q).sum()) for n, fn in obstacles.items()}


def hip_carve(P, body_mesh, body, centres, k, gap_mm, pitch_range, yaw_deg, motor_boxes=(),
              p_step=2.5, y_step=5.0):
    """The legs cut off whole at the STL's crotch plane (v18), and the lower belly
    reshaped around them: every part of the belly that a leg sweeps through over
    hip pitch pitch_range (deg, + = leg back) x hip yaw +-yaw_deg, grown by the gap,
    is carved away.  The legs keep their STL shape (a flat lid on top); the belly
    gets pockets around the leg tops.  motor_boxes: {side: [(lo, hi, kind)]} native
    boxes of the walled hip motor cases; a "pitch" case turns with the hip yaw, so
    only the hip pitch sweep is checked against it, a "yaw" case against the whole
    sweep; the leg must reach neither.  Returns dict(legs, upper,
    leg_solid, belly_solid, carve_depth (native, above the crotch), motor_hit)."""
    from skimage import measure
    zc = P.CUT_LEG_TOP
    g = gap_mm / k
    k1 = min(int((zc + 8.0 - body.lo[2]) / body.pitch) + 1, body.grid.shape[2])
    sl = (slice(None), slice(None), slice(0, k1))
    X, Y, Z = _grid_points(body, sl)
    sub = body.grid[sl]
    side = {"l": X > 0, "r": X < 0}
    pitches = np.arange(pitch_range[0], pitch_range[1] + 1e-9, p_step)
    yaws = np.arange(-yaw_deg, yaw_deg + 1e-9, y_step)
    shape = sub.shape
    carve = np.zeros(shape, bool)
    out = {"legs": {}, "leg_solid": {}}
    hit = 0
    inbox = lambda q, lo, hi: int(np.all((q >= lo) & (q <= hi), axis=1).sum())
    for sd, c in centres.items():
        leg = sub & side[sd] & (Z < zc)
        gl = np.zeros_like(body.grid)
        gl[sl] = leg
        out["leg_solid"][sd] = Solid(gl, body.lo, body.pitch)
        pts = np.c_[X[leg], Y[leg], Z[leg]]
        pts = pts[pts[:, 2] > zc - 4.0]                 # only the top can reach the belly
        for p_ in pitches:
            q0 = _rot_pitch(pts, c, p_)
            for lo, hi, kind in motor_boxes.get(sd, []):
                if kind == "pitch":
                    hit += inbox(q0, lo, hi)
            for y_ in yaws:
                q = _rot_yaw(q0, c, y_)
                for lo, hi, kind in motor_boxes.get(sd, []):
                    if kind == "yaw":
                        hit += inbox(q, lo, hi)
                idx = np.floor((q - body.lo) / body.pitch).astype(int)
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                carve[tuple(idx[ok].T)] = True
    carve = ndimage.binary_closing(carve, iterations=2)
    carve = ndimage.binary_dilation(carve, iterations=max(1, int(round(g / body.pitch))))
    legs_any = sub & (Z < zc)
    carve &= ~legs_any
    bsub = sub & ~legs_any & ~carve
    belly = body.grid.copy()
    belly[sl] = bsub
    out["belly_solid"] = Solid(belly, body.lo, body.pitch)
    cz = Z[carve & sub]
    out["carve_depth"] = float(cz.max() - zc) if len(cz) else 0.0
    out["motor_hit"] = hit
    # meshes: the legs whole below the crotch with a lid; the belly's STL skin less what
    # lies in the carve, plus the carved pocket walls (marching cubes of the carved belly)
    sdf = (ndimage.distance_transform_edt(~carve) - ndimage.distance_transform_edt(carve)) * body.pitch
    sdf = ndimage.gaussian_filter(sdf, 1.5)                 # no voxel steps along the cut
    sdf_at = lambda v: ndimage.map_coordinates(sdf, ((v - body.lo) / body.pitch - 0.5).T, order=1, mode="nearest")
    upper = []
    for sd in centres:
        sgn = 1 if sd == "l" else -1
        half = body_mesh.slice_plane([0, 0, 0], [sgn, 0, 0])
        lg = half.slice_plane([0, 0, zc], [0, 0, -1])
        lid = cap(outline(lg, zc - 1e-3), zc, up=True)
        out["legs"][sd] = trimesh.util.concatenate([lg, lid])
        up = half.slice_plane([0, 0, zc], [0, 0, 1])
        upper.append(slice_field(up, lambda v: np.where(v[:, 2] < zc + 8.0, sdf_at(v), 1.0)))
    occ = ndimage.gaussian_filter(np.pad(bsub, 1).astype(float), 1.0)
    v, f, _, _ = measure.marching_cubes(occ, 0.5)
    v = (v - 1 + 0.5) * body.pitch + body.lo
    walls = trimesh.Trimesh(v, f, process=False)
    cen = walls.triangles_center
    near = np.abs(sdf_at(cen)) < 2.5 * body.pitch
    deep = body.inside(cen + [0, 0, 0]) & ndimage.binary_erosion(body.grid, iterations=2)[
        tuple(np.clip(np.floor((cen - body.lo) / body.pitch).astype(int), 0, np.array(body.grid.shape) - 1).T)]
    walls.update_faces(near & deep & (cen[:, 2] > zc - 0.05))
    walls.remove_unreferenced_vertices()
    out["upper"] = trimesh.util.concatenate(upper + [walls])
    return out


def hip_socket(P, body_mesh, body, centres, k, gap_mm, pitch_range, yaw_deg, z_split, pitch_cases,
               yaw_boxes=None, p_step=2.5, y_step=5.0, leg_polys=None, leg_bottom=None):
    """v19, after images-modeling/PartsAssemblyV2: the torso ends in a flat bottom at
    z_split (just under the hip pitch motors) with a socket for each leg; each leg is
    its own STL section carried up from the crotch to that bottom (so it grows by
    z_split - crotch) and closed inside the torso by a dome on a sphere about the hip
    pivot (where hip yaw and pitch meet), so the hidden top turns in place.  The leg
    is hollowed where its own hip pitch motor sits (the motor turns with the hip yaw,
    so only the pitch sweep counts).  The torso is carved by the legs' whole sweep
    (pitch_range x +-yaw_deg) grown by the gap: the sockets and the round cavity
    above them.  pitch_cases: {side: (R, shaft)} of the walled hip pitch cases;
    yaw_boxes: {side: (lo, hi)} of the hip yaw cases (fixed in the torso).
    leg_polys / leg_bottom (v21): each leg is one straight column of that section from
    leg_bottom up (no STL leg below it), so the leg has no step.
    Returns dict(legs, upper, leg_solid, belly_solid, r_dome, socket_gap, slot_skin,
    motor_hit)."""
    from matplotlib.path import Path
    from skimage import measure
    import shapely.geometry as sg
    zj, zs = P.CUT_LEG_TOP - 0.5, z_split
    g = gap_mm / k
    pts_s, _ = trimesh.sample.sample_surface(body_mesh, 300000, seed=4)
    poly, rdome = {}, {}
    for sd, c in centres.items():
        sgn = 1 if sd == "l" else -1
        if leg_polys:
            poly[sd] = leg_polys[sd]
        else:
            ring = pts_s[(np.abs(pts_s[:, 2] - zj) < 0.15) & (sgn * pts_s[:, 0] > 0)]
            poly[sd] = sg.MultiPoint(ring[:, :2]).convex_hull
        ex = np.array(poly[sd].exterior.coords)
        rdome[sd] = float(np.sqrt(((ex - c[:2]) ** 2).sum(1)).max() ** 2 + (c[2] - zs) ** 2) ** 0.5 + 0.05
    ztop = max(c[2] + rdome[sd] for sd, c in centres.items()) + 1.0
    k1 = min(int((ztop - body.lo[2]) / body.pitch) + 1, body.grid.shape[2])
    sl = (slice(None), slice(None), slice(0, k1))
    X, Y, Z = _grid_points(body, sl)
    sub = body.grid[sl]
    shape = sub.shape
    xy = np.c_[X[:, :, 0].ravel(), Y[:, :, 0].ravel()]
    side = {"l": X > 0, "r": X < 0}
    pitches = np.arange(pitch_range[0], pitch_range[1] + 1e-9, p_step)
    yaws = np.arange(-yaw_deg, yaw_deg + 1e-9, y_step)
    lo_c, hi_c = case_extent(P, walls=False)          # the bare case turns inside the leg top
    out = {"legs": {}, "leg_solid": {}, "r_dome": rdome, "slot_skin": {}, "socket_gap": {}}
    carve = np.zeros(shape, bool)
    legs_any = np.zeros(shape, bool)
    hit = 0
    for sd, c in centres.items():
        inpoly = Path(np.array(poly[sd].exterior.coords)).contains_points(xy).reshape(shape[:2])[:, :, None]
        D = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        if leg_polys:                  # one column from the leg's bottom
            leg = (inpoly & (Z < zs)) | (inpoly & (Z >= zs) & (D <= rdome[sd]))
        else:
            leg = (sub & side[sd] & (Z < zj)) | (inpoly & (Z >= zj) & (Z < zs)) | (inpoly & (Z >= zs) & (D <= rdome[sd]))
        # hollow for the hip pitch motor: leg voxels that meet the walled case (+ gap) at some pitch
        R, shaft = pitch_cases[sd]
        top = leg & (Z > zj - 2.0)
        idx = np.argwhere(top)
        q0 = np.c_[X[top], Y[top], Z[top]]
        slot = np.zeros(len(q0), bool)
        for p_ in pitches:
            l_ = (_rot_pitch(q0, c, p_) - shaft) @ R.T * k
            slot |= np.all((l_ >= lo_c - gap_mm) & (l_ <= hi_c + gap_mm), axis=1)
        skin = leg & ~ndimage.binary_erosion(leg)
        sm = np.zeros(shape, bool)
        sm[tuple(idx[slot].T)] = True
        out["slot_skin"][sd] = float((sm & skin & (Z < zs)).sum() * (body.pitch * k) ** 2)   # mm^2 of visible skin
        leg &= ~sm
        legs_any |= leg
        gl = np.zeros_like(body.grid)
        gl[sl] = leg
        out["leg_solid"][sd] = Solid(gl, body.lo, body.pitch)
        pts = np.c_[X[leg], Y[leg], Z[leg]]
        pts = pts[pts[:, 2] > zj - 4.0]
        for p_ in pitches:
            q1 = _rot_pitch(pts, c, p_)
            for y_ in yaws:
                q = _rot_yaw(q1, c, y_)
                if yaw_boxes and sd in yaw_boxes:
                    hit += int(np.all((q >= yaw_boxes[sd][0]) & (q <= yaw_boxes[sd][1]), axis=1).sum())
                ii = np.floor((q - body.lo) / body.pitch).astype(int)
                ok = np.all((ii >= 0) & (ii < shape), axis=1)
                carve[tuple(ii[ok].T)] = True
    carve = ndimage.binary_closing(carve, iterations=2)
    carve = ndimage.binary_dilation(carve, iterations=max(1, int(round(g / body.pitch))))
    carve &= ~legs_any
    tsub = sub & (Z >= zs) & ~carve & ~legs_any
    belly = body.grid.copy()
    belly[sl] = tsub
    out["belly_solid"] = Solid(belly, body.lo, body.pitch)
    out["motor_hit"] = hit
    # socket gap at the torso bottom: how far the hole reaches past the leg, front / back /
    # outer side, and how thin the torso wall left outside it is
    kz = int((zs + 1.5 * body.pitch - body.lo[2]) / body.pitch)
    hole, bod = carve[:, :, kz], sub[:, :, kz]
    for sd, c in centres.items():
        sgn = 1 if sd == "l" else -1
        mine = hole & (sgn * X[:, :, 0] > 0)
        hx, hy = X[:, :, 0][mine], Y[:, :, 0][mine]
        b = poly[sd].bounds
        wall = ndimage.distance_transform_edt(bod & ~hole)
        rim = ndimage.binary_dilation(mine) & bod & ~hole
        out["socket_gap"][sd] = dict(front=(b[1] - hy.min()) * k, back=(hy.max() - b[3]) * k,
                                     outer=((hx.max() - b[2]) if sgn > 0 else (b[0] - hx.min())) * k,
                                     broken=bool((mine & ~bod).any()),
                                     wall=float(wall[rim].max()) * body.pitch * k if rim.any() else 0.0)
    # meshes
    sdf = (ndimage.distance_transform_edt(~carve) - ndimage.distance_transform_edt(carve)) * body.pitch
    sdf = ndimage.gaussian_filter(sdf, 1.5)
    sdf_at = lambda v: ndimage.map_coordinates(sdf, ((v - body.lo) / body.pitch - 0.5).T, order=1, mode="nearest")
    upper = []
    for sd, c in centres.items():
        sgn = 1 if sd == "l" else -1
        half = body_mesh.slice_plane([0, 0, 0], [sgn, 0, 0])
        r = rdome[sd]
        z0 = leg_bottom[sd] if leg_polys else zj
        stl_leg = (cap(poly[sd], z0, up=False) if leg_polys
                   else half.slice_plane([0, 0, zj], [0, 0, -1]))
        pr = trimesh.creation.extrude_polygon(poly[sd], c[2] + r - z0)
        pr.apply_translation([0, 0, z0])
        pr.update_faces(np.abs(pr.face_normals[:, 2]) < 0.5)            # side walls only
        pr.remove_unreferenced_vertices()
        dist = lambda v, c=c: np.linalg.norm(v - c, axis=1)
        walls = slice_field(pr, lambda v, r=r: np.where(v[:, 2] < zs, 1.0, r - dist(v)))
        dome = trimesh.creation.icosphere(subdivisions=5, radius=r)
        dome.apply_translation(c)
        dome = dome.slice_plane([0, 0, zs], [0, 0, 1])
        ex = np.array(poly[sd].exterior.coords)
        if poly[sd].exterior.is_ccw is False:
            ex = ex[::-1]
        for a, b_ in zip(ex[:-1], ex[1:]):                               # keep the dome inside the section
            n = np.array([-(b_[1] - a[1]), b_[0] - a[0], 0.0])
            dome = dome.slice_plane([a[0], a[1], 0], n)
        out["legs"][sd] = trimesh.util.concatenate([stl_leg, walls, dome])
        up = half.slice_plane([0, 0, zs], [0, 0, 1])
        upper.append(slice_field(up, lambda v: np.where(v[:, 2] < ztop, sdf_at(v), 1.0)))
    plate = cap(outline(body_mesh, zs + 1e-3), zs, up=False)
    upper.append(slice_field(plate, sdf_at))
    occ = ndimage.gaussian_filter(np.pad(tsub, 1).astype(float), 1.0)
    v, f, _, _ = measure.marching_cubes(occ, 0.5)
    v = (v - 1 + 0.5) * body.pitch + body.lo
    cw = trimesh.Trimesh(v, f, process=False)
    cen = cw.triangles_center
    near = np.abs(sdf_at(cen)) < 2.5 * body.pitch
    deep = ndimage.binary_erosion(body.grid, iterations=2)[
        tuple(np.clip(np.floor((cen - body.lo) / body.pitch).astype(int), 0, np.array(body.grid.shape) - 1).T)]
    cw.update_faces(near & deep & (cen[:, 2] > zs + 0.05))
    cw.remove_unreferenced_vertices()
    out["upper"] = trimesh.util.concatenate(upper + [cw])
    return out


def fit_torso_bottom(body_mesh, zs, z_up, leg_polys, margin, n=240, rings=24):
    """v21: the torso below z_up re-lofted so its bottom at zs covers both legs with
    `margin` all round (the legs then come out from under it, as in PartsAssemblyV2,
    instead of standing wider than a narrowing belly).  Each height's outline blends
    (smoothstep) from the STL's own section at z_up to the bottom outline; above z_up
    the STL is untouched; below zs the STL legs stay in the mesh (the voxel solids need
    them; the drawn legs are built on their own).  Returns the new body mesh and the
    bottom outline."""
    import shapely.geometry as sg
    from shapely.ops import unary_union
    top = outline(body_mesh, z_up)
    bot0 = outline(body_mesh, zs)
    legs = unary_union(list(leg_polys.values())).convex_hull.buffer(margin)
    bottom = unary_union([bot0, legs]).convex_hull
    c = np.array(top.centroid.coords[0])
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def radii(poly):
        out = []
        for a in ang:
            ray = sg.LineString([c, c + 200 * np.array([np.cos(a), np.sin(a)])])
            hit = ray.intersection(poly.exterior)
            pts = np.array([g.coords[0] for g in getattr(hit, "geoms", [hit])]) if not hit.is_empty else c[None]
            out.append(np.linalg.norm(pts - c, axis=1).max())
        return np.array(out)
    rt, rb = radii(top), radii(bottom)
    zz = np.linspace(zs, z_up, rings)
    t = (zz - zs) / (z_up - zs)
    sm = t * t * (3 - 2 * t)
    V = []
    for z, w in zip(zz, sm):
        r = rb + (rt - rb) * w
        V.append(np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.full(n, z)])
    V = np.vstack(V)
    F = []
    for i in range(rings - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    band = trimesh.Trimesh(V, np.array(F), process=False)
    if band.face_normals[:, :2].mean(axis=0) @ (band.triangles_center[:, :2] - c).mean(axis=0) < 0:
        band.invert()
    upper = body_mesh.slice_plane([0, 0, z_up], [0, 0, 1])
    lower = body_mesh.slice_plane([0, 0, zs], [0, 0, -1])     # the STL legs: kept so the voxel solids
    return trimesh.util.concatenate([upper, band, lower]), bottom   # reach the leg tops and motors
