#!/usr/bin/env python3
"""Give Haetburi real feet: straight legs ending in Doraemon-style pads.

The source solid ends each leg in a rounded tip that tapers from 32 mm deep at
the crotch to 11 mm at the sole (21 x 23 mm at z = 10).  This replaces everything below LEG_STRAIGHT_Z
with a straight extrusion of the leg cross-section at that height, and sets
each leg on a flat, rounded, forward-reaching foot pad.  Total height and the
leg length (sole to crotch) stay exactly as in the source.

All numbers are native STL millimetres (300 mm tall character).
Frame: -y is the face direction, +z up, sole at z = 0.

    python3 tools/make_feet.py        -> modelings/haetburi_feet_v1.stl
"""
import os

import numpy as np
import shapely.geometry as sg
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "modelings", "haetburi_grip_20mm_300mm.stl")
OUT = os.path.join(ROOT, "modelings", "haetburi_feet_v1.stl")

# ------------------------------------------------------------------ the leg
LEG_STRAIGHT_Z = 10.0   # below this the leg is a straight prism of this section
LEG_X_LIMIT = 25.0      # |x| beyond this at leg height is the tail, not a leg

# ----------------------------------------------------------------- the foot
# Doraemon's feet: a flat-bottomed oval bun, clearly wider and longer than the
# leg, reaching forward past the toes of the leg and a little behind the heel.
FOOT_LENGTH = 42.0      # fore-aft   (leg is 25 deep)
FOOT_WIDTH = 27.0       # sideways   (leg is 21 wide)
FOOT_HEIGHT = 9.0       # sole to top of the dome
FOOT_FORWARD = 3.5      # pad centre ahead of the leg centre
FOOT_OUTWARD = 0.5      # pad centre outboard of the leg centre
FOOT_TOE_OUT = 5.0      # degrees, toes turned outward like the character stands
MIN_RIM = 1.5           # the leg must sit this far inside the pad outline (plan view)
FOOT_BULGE = 0.30       # how far below the widest line the flat sole is cut
                        # (0 = half-ellipsoid, larger = rounder side bulge)


def leg_sections(mesh, z):
    """Cross-section polygons of the two legs at height z, sorted -x then +x."""
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    polys = [sg.Polygon(l[:, :2]) for l in s.discrete if len(l) > 3]
    legs = [p for p in polys if p.area > 50 and abs(p.centroid.x) < LEG_X_LIMIT]
    if len(legs) != 2:
        raise RuntimeError(f"expected 2 leg sections at z={z}, got {len(legs)}")
    return sorted(legs, key=lambda p: p.centroid.x)


def _sections(mesh, z):
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    return [sg.Polygon(l[:, :2]) for l in s.discrete if len(l) > 3 and
            sg.Polygon(l[:, :2]).area > 50]


def straight_leg(poly, z_bottom, z_top):
    prism = trimesh.creation.extrude_polygon(poly.buffer(0), z_top - z_bottom)
    prism.apply_translation([0, 0, z_bottom])
    return prism


def foot_pad(cx, cy, side):
    """Oval bun: an ellipsoid with its bottom sliced flat, toe-out rotated."""
    c = FOOT_HEIGHT / (1.0 + FOOT_BULGE)             # vertical semi-axis
    a, b = FOOT_WIDTH / 2.0, FOOT_LENGTH / 2.0       # widest half-width / half-length
    pad = trimesh.creation.icosphere(subdivisions=5)
    pad.apply_scale([a, b, c])
    pad.apply_translation([0, 0, FOOT_BULGE * c])
    pad = pad.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    # toes point to -y; turn them outward (left foot is +x in this frame)
    ang = np.deg2rad(FOOT_TOE_OUT) * (-side)
    pad.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
    pad.apply_translation([cx + side * FOOT_OUTWARD, cy - FOOT_FORWARD, 0])
    return pad


def build():
    src = trimesh.load(SRC)
    upper = src.slice_plane([0, 0, LEG_STRAIGHT_Z], [0, 0, 1], cap=True)
    parts = [upper]
    feet = []
    for side, poly in zip((-1, +1), leg_sections(src, LEG_STRAIGHT_Z)):
        # overlap the prism into both the foot and the upper body so the union is solid
        parts.append(straight_leg(poly, FOOT_HEIGHT * 0.5, LEG_STRAIGHT_Z + 0.5))
        f = foot_pad(poly.centroid.x, poly.centroid.y, side)
        feet.append(f)
        parts.append(f)
    out = trimesh.boolean.union(parts, engine="manifold")
    return out, feet


def main():
    mesh, feet = build()
    mesh.export(OUT)
    gap = trimesh.proximity.closest_point(feet[1], feet[0].vertices)[1].min()
    # the leg has to stand on the pad, not hang over its edge
    z_wide = FOOT_BULGE * FOOT_HEIGHT / (1.0 + FOOT_BULGE)   # widest line of the pad
    sole = [p for p in _sections(mesh, z_wide)]
    legs = leg_sections(mesh, FOOT_HEIGHT + 1.5)
    rims = []
    for leg in legs:
        pad = max(sole, key=lambda p: p.intersection(leg).area)
        rims.append(pad.exterior.distance(leg) if pad.contains(leg) else
                    -leg.difference(pad).area)
    rim = min(rims)
    b = mesh.bounds
    print(f"wrote {OUT}")
    print(f"  watertight {mesh.is_watertight}, {len(mesh.faces)} faces, "
          f"height {b[1][2]-b[0][2]:.1f} mm")
    for name, f in zip(("right foot (-x)", "left foot (+x)"), feet):
        fb = f.bounds
        print(f"  {name}: x {fb[0][0]:6.1f}..{fb[1][0]:6.1f}  y {fb[0][1]:6.1f}..{fb[1][1]:6.1f}"
              f"  z 0..{fb[1][2]:.1f}")
    print(f"  closest gap between the two feet: {gap:.1f} mm")
    print(f"  leg-to-sole-edge rim: {rim:.1f} mm "
          f"{'OK' if rim >= MIN_RIM else '<- the leg overhangs the pad' if rim < 0 else '<- too thin'}")


if __name__ == "__main__":
    main()
