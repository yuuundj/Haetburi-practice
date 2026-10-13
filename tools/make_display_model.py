#!/usr/bin/env python3
"""Turn modelings/haetburi_display_parts.stl into a base character model.

The new STL (images-modeling/Haetburi_display.png, 120 mm tall) has no tail and
carries its own, smaller display.  This script
  1. keeps the 8 real parts (head, backpack, body + legs, display, 2 arms, 2 feet)
     and drops ~33 500 stray one-triangle fragments,
  2. takes the tail from the old character (modelings/haetburi_grip_20mm_300mm.stl),
     scales it to the new model (120/300), turns it so it grows backward out of the
     upper middle of the bottom (just under the backpack) and curls to the left,
  3. removes the STL's own display and puts back one the size the robot already
     uses (210.5 x 142.3 x 11.5 mm at a 500 mm character), in the same spot,
     lowered until it clears the head and body,
  4. bends both arms outward (shoulders fixed, fists moved most) so the fists hold
     the two ends of the wider display the way they held the STL's own one,

    python3 tools/make_display_model.py
Out (native 120 mm frame, -y = face direction, z up from the sole):
    modelings/haetburi_display_tail.stl       character + tail, no display  (base model)
    modelings/haetburi_display_assembled.stl  the same + the resized display (for review)
"""
import os

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, "modelings", "haetburi_display_parts.stl")
OLD = os.path.join(ROOT, "modelings", "haetburi_grip_20mm_300mm.stl")
OUT_BASE = os.path.join(ROOT, "modelings", "haetburi_display_tail.stl")
OUT_ASM = os.path.join(ROOT, "modelings", "haetburi_display_assembled.stl")

NEW_HEIGHT, OLD_HEIGHT = 120.0, 300.0
CHAR_HEIGHT_MM = 500.0                    # the robot builds the character this tall
DISPLAY_MM = (210.5, 142.3, 11.5)         # width, height, thickness at CHAR_HEIGHT_MM
DISPLAY_CLEAR_MM = 3.0

# arms: fists reach forward from the shoulders (y ~ +2) to y ~ -27.  Vertices
# behind ARM_FIXED_Y stay put, vertices in front of ARM_FULL_Y move the whole
# way, with a smooth blend in between.  The STL's fist sat FIST_EDGE_OFFSET
# outboard of its display's edge; keep that grip on the wider display.
ARM_FIXED_Y, ARM_FULL_Y = -2.0, -16.0
FIST_Y = -16.0                               # fist = arm vertices in front of this
FIST_EDGE_OFFSET = None                      # measured from the STL when None

PART_NAMES = ["head", "backpack", "body_legs", "display", "arm_a", "arm_b", "foot_a", "foot_b"]

# old tail: the part of the old solid beside the left hip (x_stl > 28) between z 14 and 52,
# largest connected piece = the tail (the rest are slivers of belly and backpack strap)
OLD_TAIL_SLAB = (14.0, 52.0, 28.0)
# where the tail leaves the body: the upper middle of the bottom, just under the
# backpack (backpack starts at z 21.6, the root is 4.6 tall)
TAIL_ROOT_Z = 18.0
TAIL_EMBED = 0.6                          # sink the tail root this far into the body


def main_parts(mesh, n=8):
    parts = sorted(mesh.split(only_watertight=False), key=lambda p: -p.area)
    big, rest = parts[:n], parts[n:]
    print(f"kept {n} parts ({sum(len(p.faces) for p in big)} faces), "
          f"dropped {len(rest)} fragments ({sum(len(p.faces) for p in rest)} faces)")
    return dict(zip(PART_NAMES, big))


def old_tail():
    """The old tail, scaled to the new model.  Its root is the cut at x_stl = 28
    (facing -x); it grows toward +x and curls toward +y (back)."""
    old = trimesh.load(OLD)
    zlo, zhi, x0 = OLD_TAIL_SLAB
    slab = old.slice_plane([0, 0, zlo], [0, 0, 1], cap=True).slice_plane([0, 0, zhi], [0, 0, -1], cap=True)
    side = slab.slice_plane([x0, 0, 0], [1, 0, 0], cap=True)
    tail = max(side.split(only_watertight=False), key=lambda c: c.area)
    tail.apply_scale(NEW_HEIGHT / OLD_HEIGHT)
    return tail


def attach(tail, body):
    """Swap x and y so the tail grows backward (+y) and curls to the robot's left
    (+x), then put the root at the middle of the back of the bottom at
    TAIL_ROOT_Z, sunk TAIL_EMBED into the body."""
    swap = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    tail.apply_transform(swap)                         # a mirror: trimesh flips the winding back
    v = tail.vertices
    root = v[v[:, 1] < v[:, 1].min() + 0.3]
    c = root.mean(axis=0)
    b = body.vertices
    back = b[(np.abs(b[:, 2] - TAIL_ROOT_Z) < 0.5) & (np.abs(b[:, 0]) < 2.0)][:, 1].max()
    shift = np.array([-c[0], (back - TAIL_EMBED) - root[:, 1].min(), TAIL_ROOT_Z - c[2]])
    tail.apply_translation(shift)
    return shift, back


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def widen_arms(parts, half_width):
    """Bend each arm outward so its fist grips the display end at +-half_width."""
    old_edge = parts["display"].bounds[1][0]                 # the STL display's own edge
    out = {}
    for k in ("arm_a", "arm_b"):
        arm = parts[k].copy()
        v = arm.vertices
        side = np.sign(v[:, 0].mean())
        fist = v[v[:, 1] < FIST_Y]
        fist_cx = np.abs(fist[:, 0]).mean()
        offset = fist_cx - old_edge if FIST_EDGE_OFFSET is None else FIST_EDGE_OFFSET
        shift = (half_width + offset) - fist_cx
        wgt = _smoothstep((ARM_FIXED_Y - v[:, 1]) / (ARM_FIXED_Y - ARM_FULL_Y))
        v = v.copy()
        v[:, 0] += side * shift * wgt
        arm.vertices = v
        out[k] = (arm, shift, offset)
    return out


def display_box(parts, char_height_mm=CHAR_HEIGHT_MM, display_mm=DISPLAY_MM):
    s = NEW_HEIGHT / char_height_mm                      # mm -> native units
    w, h, t = (d * s for d in display_mm)
    old = parts["display"]
    cx, cy, cz = old.bounds.mean(axis=0)
    others = np.vstack([parts[k].vertices for k in ("head", "body_legs", "backpack", "foot_a", "foot_b")])
    g = DISPLAY_CLEAR_MM * s
    while True:
        hit = ((np.abs(others[:, 0] - cx) < w / 2 + g) & (np.abs(others[:, 1] - cy) < t / 2 + g)
               & (np.abs(others[:, 2] - cz) < h / 2 + g))
        if not hit.any():
            break
        cz -= 0.1
    box = trimesh.creation.box(extents=(w, t, h))
    box.apply_translation([cx, cy, cz])
    return box, (cx, cy, cz), (w, h, t), old.extents


def build(char_height_mm=CHAR_HEIGHT_MM, log=print, display_mm=DISPLAY_MM):
    """The character with the tail on and the fists on a DISPLAY_MM display, for a
    character char_height_mm tall (the display keeps its size in mm, so the arms
    bend to fit it at any scale).  Returns (parts without the STL's display,
    display box, its centre and (w, h, t) in native units)."""
    parts = main_parts(trimesh.load(NEW))
    tail = old_tail()
    shift, back = attach(tail, parts["body_legs"])
    log(f"tail: scaled x{NEW_HEIGHT/OLD_HEIGHT:.2f}, turned to grow backward, root centred at x 0, "
        f"z {TAIL_ROOT_Z}, sunk {TAIL_EMBED} into the back (surface y {back:.1f}); "
        f"tail spans x {tail.bounds[0][0]:.1f}..{tail.bounds[1][0]:.1f}, "
        f"y {tail.bounds[0][1]:.1f}..{tail.bounds[1][1]:.1f}, z {tail.bounds[0][2]:.1f}..{tail.bounds[1][2]:.1f}; "
        f"winding ok {tail.is_winding_consistent}, volume {tail.volume:+.0f}")
    box, c, (w, h, t), old_ext = display_box(parts, char_height_mm, display_mm)
    s = char_height_mm / NEW_HEIGHT
    log(f"display: STL's own {old_ext[0]*s:.0f} x {old_ext[2]*s:.0f} x {old_ext[1]*s:.0f} mm replaced by "
        f"{w*s:.1f} x {h*s:.1f} x {t*s:.1f} mm; centre ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) native, "
        f"bottom edge {(c[2]-h/2)*s:.0f} mm above the sole at a {char_height_mm:.0f} mm character")
    arms = widen_arms(parts, w / 2)
    for k, (arm, shift, off) in arms.items():
        parts[k] = arm
        log(f"{k}: fist moved {shift:+.2f} outward ({shift*s:+.0f} mm), gripping {off:+.2f} past the edge")
    parts = {k: p for k, p in parts.items() if k != "display"}
    parts["tail"] = tail
    return parts, box, c, (w, h, t)


def main():
    parts, box, c, (w, h, t) = build()
    s = CHAR_HEIGHT_MM / NEW_HEIGHT
    tail = parts["tail"]
    character = list(parts.values())
    base = trimesh.util.concatenate(character)
    base.export(OUT_BASE)
    trimesh.util.concatenate(character + [box]).export(OUT_ASM)
    print("wrote", os.path.relpath(OUT_BASE, ROOT), "and", os.path.relpath(OUT_ASM, ROOT),
          f"({len(base.faces)} faces, height {base.extents[2]:.1f})")
    # tail vs legs / feet: any overlap?
    for k in ("foot_a", "foot_b", "backpack", "body_legs"):
        d = trimesh.proximity.closest_point(parts[k], tail.vertices)[1].min()
        print(f"  closest tail-to-{k} distance {d:.2f} native ({d*s:.1f} mm)")


if __name__ == "__main__":
    main()
