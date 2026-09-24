"""Cut the Haetburi solid into link meshes, one recipe per layout.

All pieces are returned in the native STL frame (mm).
"""
import numpy as np
import shapely.geometry as sg
import trimesh

import haetburi_params as P


def _slab(mesh, z_lo, z_hi):
    m = mesh
    if z_lo is not None:
        m = m.slice_plane([0, 0, z_lo], [0, 0, 1], cap=True)
    if z_hi is not None:
        m = m.slice_plane([0, 0, z_hi], [0, 0, -1], cap=True)
    return m


def _leg_sections(mesh, z):
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    polys = [sg.Polygon(l[:, :2]) for l in s.discrete if len(l) > 3]
    legs = [p for p in polys if p.area > 50 and abs(p.centroid.x) < P.LEG_X_LIMIT]
    if len(legs) != 2:
        raise RuntimeError(f"expected 2 leg sections at z={z}, got {len(legs)}")
    return sorted(legs, key=lambda p: p.centroid.x)


def _split_legs(slab):
    legs, strays = [], []
    for comp in slab.split(only_watertight=False):
        (legs if abs(comp.centroid[0]) < P.LEG_X_LIMIT else strays).append(comp)
    if len(legs) != 2:
        raise RuntimeError(f"expected 2 leg components, got {len(legs)}")
    legs.sort(key=lambda c: c.centroid[0])            # x_stl < 0 is the robot's right
    return legs, strays


def _with_meta(mesh, centre, bottom):
    mesh.metadata["centre_native"] = centre
    mesh.metadata["bottom_native"] = bottom
    return mesh


# --------------------------------------------------------- v1 / v2: exposed
def segment_exposed():
    """Original tapering legs below the crotch; the ankle stack goes under them."""
    src = trimesh.load(P.SRC_STL)
    (leg_r, leg_l), strays = _split_legs(_slab(src, None, P.CUT_LEG_TOP))
    pelvis = _slab(src, P.CUT_LEG_TOP, P.CUT_WAIST)
    out = {"leg_l": leg_l, "leg_r": leg_r,
           "pelvis": trimesh.util.concatenate([pelvis] + strays) if strays else pelvis,
           "torso": _slab(src, P.CUT_WAIST, None)}
    for k in ("leg_l", "leg_r"):
        c = out[k].centroid
        _with_meta(out[k], (c[0], c[1]), 0.0)
    return out


# ----------------------------------------------------------- v4: foot_both
def segment_foot_both():
    """Leg made straight below LEG_STRAIGHT_Z, starting at LEG_BOTTOM_NATIVE."""
    src = trimesh.load(P.SRC_STL)
    (flare_r, flare_l), strays = _split_legs(_slab(src, P.LEG_STRAIGHT_Z, P.CUT_LEG_TOP))
    out = {}
    for key, fl, sec in zip(("leg_r", "leg_l"), (flare_r, flare_l),
                            _leg_sections(src, P.LEG_STRAIGHT_Z)):
        h = P.LEG_STRAIGHT_Z + 0.5 - P.LEG_BOTTOM_NATIVE
        prism = trimesh.creation.extrude_polygon(sec.buffer(0), h)
        prism.apply_translation([0, 0, P.LEG_BOTTOM_NATIVE])
        leg = trimesh.boolean.union([fl, prism], engine="manifold")
        c = leg.centroid
        out[key] = _with_meta(leg, (c[0], c[1]), P.LEG_BOTTOM_NATIVE)
    pelvis = _slab(src, P.CUT_LEG_TOP, P.CUT_WAIST)
    out["pelvis"] = trimesh.util.concatenate([pelvis] + strays) if strays else pelvis
    out["torso"] = _slab(src, P.CUT_WAIST, None)
    return out


# --------------------------------------------------------- v5 / v6: split
def segment_split(leg_len_mm, leg_half_mm):
    """Rounded-square leg prisms leg_len_mm long that hold the roll motor.

    boxy  (v5): prism below LEG_STRAIGHT_Z, original flare above it into the belly
    round (v6): prism all the way to the crotch; the lower belly rebuilt as a
                "shorts" hull of the belly ring and both leg tops; tail root kept
    """
    import layout as LY
    src = trimesh.load(P.SRC_STL)
    s = P.MESH_SCALE
    hx, hy = leg_half_mm
    ring = LY.leg_polygon(hy / s, hx / s, P.LEG_EXPONENT)     # x_stl = width, y_stl = depth
    bottom = P.CUT_LEG_TOP - leg_len_mm / s
    sections = _leg_sections(src, P.LEG_STRAIGHT_Z)
    out, tops = {}, []

    if P.TRUNK_BLEND_TOP is None:
        flares, strays = _split_legs(_slab(src, P.LEG_STRAIGHT_Z, P.CUT_LEG_TOP))
        prism_top = P.LEG_STRAIGHT_Z + 0.5
    else:
        flares, prism_top = (None, None), P.CUT_LEG_TOP

    for key, fl, sec in zip(("leg_r", "leg_l"), flares, sections):
        c = sec.centroid
        pts2 = ring + np.array([c.x, c.y])
        prism = trimesh.creation.extrude_polygon(sg.Polygon(pts2), prism_top - bottom)
        prism.apply_translation([0, 0, bottom])
        leg = prism if fl is None else trimesh.boolean.union([fl, prism], engine="manifold")
        out[key] = _with_meta(leg, (c.x, c.y), bottom)
        tops.append(np.c_[pts2, np.full(len(pts2), P.CUT_LEG_TOP)])

    if P.TRUNK_BLEND_TOP is None:
        pelvis = _slab(src, P.CUT_LEG_TOP, P.CUT_WAIST)
        out["pelvis"] = trimesh.util.concatenate([pelvis] + strays) if strays else pelvis
    else:
        belly = _slab(src, P.TRUNK_BLEND_TOP, P.CUT_WAIST)
        # the tail joins the body below the blend height; keep its root
        low = _slab(src, P.LEG_STRAIGHT_Z, P.TRUNK_BLEND_TOP)
        tail_root = low.slice_plane([P.TAIL_CUT_X, 0, 0], [1, 0, 0], cap=True)
        v = src.vertices
        band = v[(v[:, 2] >= P.TRUNK_BLEND_TOP) & (v[:, 2] <= P.TRUNK_BLEND_TOP + 3.0)
                 & (np.abs(v[:, 0]) < P.LEG_X_LIMIT + 15.0)]
        shorts = trimesh.convex.convex_hull(np.vstack([band] + tops))
        out["pelvis"] = trimesh.boolean.union([belly, shorts, tail_root], engine="manifold")
    out["torso"] = _slab(src, P.CUT_WAIST, None)
    return out
