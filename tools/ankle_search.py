#!/usr/bin/env python3
"""Exhaustive search for the most compact 2-DoF ankle made of two XL430s that
fits inside a Doraemon-style foot.

XL430 geometry (ROBOTIS drawing, same case as XM430): 28.5 (W) x 46.5 (H) x 34 (D).
The output shaft runs along D and sits on the W centre line, 11.25 mm from one
end of H.  A motor frame is therefore: D in [-17, 17] (+ horn), H in
[-11.25, 35.25], W in [-14.25, 14.25], shaft at the origin.

The pitch axis (y) and roll axis (x) both pass under the leg centre; the upper
axis may sit a height dz above (or below) the lower one.  Forcing them to meet
at one point never works: both cases would have to contain that point.  One
motor is fixed in the foot (lower joint); the other rides on the lower joint's
output and swings through its range inside the foot, carrying the leg.  For every orientation and every
shift along the shafts we check that the two cases never collide over the
lower joint's range, size the foot to wrap everything, and score the foot.

    python3 tools/ankle_search.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import haetburi_params as P

W, D = P.MOTOR_W, P.MOTOR_D
H_SHORT = P.MOTOR_AXIS_FROM_END
H_LONG = P.MOTOR_H - H_SHORT
HORN, WALL, CLEAR, FLOOR = P.MOTOR_HORN, P.WALL, P.CLEAR, P.FLOOR
PITCH_RANGE = P.LIM["ankle_pitch"][1]   # deg; the 500 mm robot tips over at ~19 deg
ROLL_RANGE = P.LIM["ankle_roll"][1]     # deg

# cost of making the foot bigger, per mm: height is the cheapest by request
W_LEN, W_WID, W_HGT = (P.FOOT_COST[k] for k in ("length", "width", "height"))
# the feet may not reach closer than this to the robot's centre line
LEG_Y = P.LEG_HALF_SPACING_NATIVE * P.MESH_SCALE
INNER_LIMIT = LEG_Y - 2.0   # max reach of a foot toward the other foot (from leg centre)

AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def motor_box(shaft, h_dir, shift):
    """Return (centre, R, half) of an oriented box, shaft through the origin.
    shaft: 'x' or 'y'; h_dir: signed unit vector along H; shift: slide along shaft."""
    d = AX[shaft]
    h = h_dir
    w = np.cross(d, h)
    R = np.stack([d, h, w], axis=1)                 # columns: D, H, W axes
    centre_local = np.array([shift, (H_LONG - H_SHORT) / 2.0, 0.0])
    half = np.array([(D + HORN) / 2.0, (H_LONG + H_SHORT) / 2.0, W / 2.0])
    return R @ centre_local, R, half


def rot(axis, deg):
    a = np.deg2rad(deg)
    k = AX[axis]
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def obb_overlap(a, b, margin):
    """Separating-axis test for two oriented boxes, inflated by margin/2 each."""
    ca, Ra, ha = a
    cb, Rb, hb = b
    ha, hb = ha + margin / 2, hb + margin / 2
    t = cb - ca
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for L in axes:
        n = np.linalg.norm(L)
        if n < 1e-9:
            continue
        L = L / n
        ra = np.sum(ha * np.abs(Ra.T @ L))
        rb = np.sum(hb * np.abs(Rb.T @ L))
        if abs(t @ L) > ra + rb:
            return False
    return True


def corners(box):
    c, R, h = box
    s = np.array(list(itertools.product((-1, 1), repeat=3)))
    return c + (s * h) @ R.T


def evaluate(lower, lo_h, up_h, lo_shift, up_shift, dz):
    lo_shaft, up_shaft = ("y", "x") if lower == "pitch" else ("x", "y")
    rng = PITCH_RANGE if lower == "pitch" else ROLL_RANGE
    lo = motor_box(lo_shaft, lo_h, lo_shift)
    c, R, h = motor_box(up_shaft, up_h, up_shift)
    up0 = (c + np.array([0, 0, dz]), R, h)
    pts = [corners(lo)]
    env = [corners((lo[0], lo[1], lo[2] + WALL))]
    for ang in np.linspace(-rng, rng, 13):
        Rr = rot(lo_shaft, ang)
        up = (Rr @ up0[0], Rr @ up0[1], up0[2])
        if obb_overlap(lo, up, CLEAR):
            return None
        pts.append(corners(up))
    for ang in np.linspace(-rng, rng, 31):            # finer sweep for the foot shell
        Rr = rot(lo_shaft, ang)
        env.append(corners((Rr @ up0[0], Rr @ up0[1], up0[2] + WALL)))
    pts = np.vstack(pts)
    env = np.vstack(env)
    lo_pt, hi_pt = pts.min(axis=0) - WALL, pts.max(axis=0) + WALL
    lo_pt[2] -= FLOOR - WALL                      # sole plate instead of a wall
    if -lo_pt[1] > INNER_LIMIT:                   # reaches into the other foot
        return None
    L, Wd, Hh = hi_pt - lo_pt
    z_p = -lo_pt[2]                               # ankle centre height above the sole
    # the upper case carries the leg: its output face must reach the foot top
    cost = W_LEN * L + W_WID * Wd + W_HGT * Hh
    return dict(cost=cost, L=L, W=Wd, H=Hh, z_p=z_p, lo=lo_pt, hi=hi_pt, dz=dz, env=env,
                z0=pts[:, 2].min() - FLOOR,
                lower=lower, lo_h=lo_h, up_h=up_h, lo_shift=lo_shift, up_shift=up_shift)


def fit_bun(env, z0, n, inner_limit=INNER_LIMIT):
    """Smallest superellipsoid bun with a flat sole at z0 wrapping `env`.

    n is one exponent, or (p, q) for plan and side roundness:
        (|x-cx|/a)^p + (|y-cy|/b)^p)^(q/p) + (|z-cz|/c)^q <= 1.
    The bun may sit off-centre (cx, cy); its inner side (-y) must stay within
    inner_limit of the leg centre.  Every (cy, b, c) is tried: a taller bun lets
    the length shrink, so no dimension can be fixed before the others."""
    p, q = (n, n) if np.isscalar(n) else n
    if P.FIT_LEGACY:
        return _fit_bun_legacy(env, z0, p, inner_limit)
    best = None
    cx = (env[:, 0].min() + env[:, 0].max()) / 2.0
    dx = np.abs(env[:, 0] - cx)
    cs = np.arange(15.0, 200.0, 0.5)[:, None]            # candidate c, one per row
    for cy in np.arange(-6.0, 24.1, 1.0):
        dy = np.abs(env[:, 1] - cy)
        for b in np.arange(dy.max() + 0.25, 60.0, 0.5):
            if b - cy > inner_limit:
                break
            cz = z0 + P.FOOT_SOLE_CUT * cs
            R = np.clip(1.0 - (np.abs(env[:, 2] - cz) / cs) ** q, 0.0, None) ** (p / q)
            rem = R - (dy / b) ** p
            ok = (rem > 0).all(axis=1)
            if not ok.any():
                continue
            a = (dx / np.where(rem > 0, rem, np.nan) ** (1.0 / p)).max(axis=1)
            L, Wd, Hh = 2 * a, 2 * b, (cz + cs - z0)[:, 0]
            cost = W_LEN * L + W_WID * Wd + W_HGT * Hh
            cost[~ok] = np.inf
            if P.FOOT_MAX_H is not None:                 # height budget (v10)
                cost[Hh > P.FOOT_MAX_H] = np.inf
            if not np.isfinite(cost).any():
                continue
            i = int(np.argmin(cost))
            if best is None or cost[i] < best["cost"]:
                best = dict(cost=cost[i], L=L[i], W=Wd, H=Hh[i], a=a[i], b=b, p=p, q=q,
                            c=float(cs[i, 0]), cx=cx, cy=cy, cz=float(cz[i, 0]))
    return best


def _fit_bun_legacy(env, z0, n, inner_limit):
    """The fitter as it was when v4_ankle_in_foot was built: for each width it
    took the first (lowest) height that fits and never tried taller buns, so the
    foot came out longer than it had to be.  Kept only to reproduce that model."""
    best = None
    cx = (env[:, 0].min() + env[:, 0].max()) / 2.0
    dx = np.abs(env[:, 0] - cx)
    for cy in np.arange(-6.0, 24.1, 1.0):
        dy = np.abs(env[:, 1] - cy)
        for b in np.arange(dy.max() + 0.25, 60.0, 0.5):
            if b - cy > inner_limit:
                break
            ry = (dy / b) ** n
            for c in np.arange(20.0, 160.0, 1.0):
                cz = z0 + P.FOOT_SOLE_CUT * c
                rem = 1.0 - ry - (np.abs(env[:, 2] - cz) / c) ** n
                if (rem <= 0).any():
                    continue
                a = (dx / rem ** (1.0 / n)).max()
                L, Wd, Hh = 2 * a, 2 * b, cz + c - z0
                cost = W_LEN * L + W_WID * Wd + W_HGT * Hh
                if best is None or cost < best["cost"]:
                    best = dict(cost=cost, L=L, W=Wd, H=Hh, a=a, b=b, c=c, cx=cx, cy=cy,
                                cz=cz, p=n, q=n)
                break
    return best


def rank_with_bun(results, n=None, top=60):
    """Re-rank the best box-cost layouts by the size of the rounded foot that
    actually has to wrap them."""
    n = (P.FOOT_PLAN_EXPONENT, P.FOOT_SIDE_EXPONENT) if n is None else n
    pool, seen = [], set()
    for r in results:
        key = (r["lower"],) + r["names"] + (r["dz"],)
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)
        if len(pool) >= top:
            break
    out = []
    for r in pool:
        f = fit_bun(r["env"], r["z0"], n)
        if f:
            out.append((f, r))
    out.sort(key=lambda fr: fr[0]["cost"])
    return out


def dirs_perp(shaft):
    out = []
    for n, v in AX.items():
        if n != shaft:
            out += [(f"+{n}", v), (f"-{n}", -v)]
    return out


def main():
    shifts = np.arange(-24, 24.1, 4.0)
    dzs = np.arange(-50, 50.1, 5.0)
    res = []
    for lower in ("pitch", "roll"):
        lo_shaft, up_shaft = ("y", "x") if lower == "pitch" else ("x", "y")
        for (ln, lh), (un, uh) in itertools.product(dirs_perp(lo_shaft), dirs_perp(up_shaft)):
            for ls, us, dz in itertools.product(shifts, shifts, dzs):
                r = evaluate(lower, lh, uh, ls, us, dz)
                if r:
                    r["names"] = (ln, un)
                    res.append(r)
    res.sort(key=lambda r: r["cost"])
    print(f"{len(res)} collision-free layouts; best per (lower joint, H directions):")
    seen, shown = set(), 0
    print(f"  {'lower':6s} {'lowerH':>6s} {'upperH':>6s} {'shiftL':>6s} {'shiftU':>6s} {'dz':>5s}"
          f" {'L':>6s} {'W':>6s} {'H':>6s} {'lowAx z':>7s} {'cost':>7s}  foot x-range, y-range")
    for r in res:
        key = (r["lower"],) + r["names"]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['lower']:6s} {r['names'][0]:>6s} {r['names'][1]:>6s} {r['lo_shift']:6.0f} "
              f"{r['up_shift']:6.0f} {r['dz']:5.0f} {r['L']:6.1f} {r['W']:6.1f} {r['H']:6.1f} {r['z_p']:7.1f} "
              f"{r['cost']:7.1f}  x[{r['lo'][0]:5.1f},{r['hi'][0]:5.1f}] y[{r['lo'][1]:5.1f},{r['hi'][1]:5.1f}]")
        shown += 1
        if shown >= 12:
            break
    return res


if __name__ == "__main__":
    main()
