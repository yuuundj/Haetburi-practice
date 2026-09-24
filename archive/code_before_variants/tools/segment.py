"""Cut the Haetburi solid into link meshes, with straight rounded-square legs.

Returns pieces in the native STL frame (mm).  The feet are not taken from the
source; tools/foot.py generates them around the ankle motors.
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


def segment(leg_len_mm, leg_half_mm, stl_path=P.SRC_STL):
    """{leg_l, leg_r, pelvis, torso} in the native frame.

    leg_*  : a straight rounded-square prism, leg_len_mm long (world mm), that
             holds the roll motor and ends at the crotch
    pelvis : the belly from TRUNK_BLEND_TOP to the waist, plus a smooth "shorts"
             piece (convex hull of the belly ring and both leg tops) so the square
             legs run straight into the body; the tail root is kept
    torso  : everything above the waist
    leg_half_mm: (half depth, half width) of the prism in world mm
    """
    import layout as LY
    src = trimesh.load(stl_path)
    s = P.MESH_SCALE
    belly = _slab(src, P.TRUNK_BLEND_TOP, P.CUT_WAIST)
    torso = _slab(src, P.CUT_WAIST, None)

    # The tail joins the body below the blend height (its root reaches x_stl ~34
    # at z ~24), so cut the tail side off that band and keep it with the pelvis.
    low = _slab(src, P.LEG_STRAIGHT_Z, P.TRUNK_BLEND_TOP)
    tail_root = low.slice_plane([P.TAIL_CUT_X, 0, 0], [1, 0, 0], cap=True)
    strays = [tail_root]

    hx, hy = leg_half_mm
    ring = LY.leg_polygon(hy / s, hx / s, P.LEG_EXPONENT)       # x_stl = width, y_stl = depth
    bottom = P.CUT_LEG_TOP - leg_len_mm / s
    out, tops = {}, []
    for key, sec in zip(("leg_r", "leg_l"), _leg_sections(src, P.LEG_STRAIGHT_Z)):
        c = sec.centroid
        pts2 = ring + np.array([c.x, c.y])
        prism = trimesh.creation.extrude_polygon(sg.Polygon(pts2), P.CUT_LEG_TOP - bottom)
        prism.apply_translation([0, 0, bottom])
        prism.metadata["centre_native"] = (c.x, c.y)
        prism.metadata["bottom_native"] = bottom
        out[key] = prism
        tops.append(np.c_[pts2, np.full(len(pts2), P.CUT_LEG_TOP)])

    # shorts: hull of the belly ring just above the blend height and the leg tops
    v = src.vertices
    band = v[(v[:, 2] >= P.TRUNK_BLEND_TOP) & (v[:, 2] <= P.TRUNK_BLEND_TOP + 3.0)
             & (np.abs(v[:, 0]) < P.LEG_X_LIMIT + 15.0)]            # leave the tail out
    shorts = trimesh.convex.convex_hull(np.vstack([band] + tops))
    out["pelvis"] = trimesh.boolean.union([belly, shorts] + strays, engine="manifold")
    out["torso"] = torso
    return out


if __name__ == "__main__":
    import layout as LY
    for name, m in segment(LY.leg_length()[0], LY.leg_section()).items():
        b = m.bounds
        print(f"{name:7s} faces={len(m.faces):7d} watertight={m.is_watertight} "
              f"bbox={np.round(b[0], 1)} .. {np.round(b[1], 1)}")
