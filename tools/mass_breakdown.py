#!/usr/bin/env python3
"""Where a variant's mass goes: printed shell per part (area x wall x rib factor x PLA),
servos, display, electronics, and the rest (brackets, sole plates, screws), next to
the built model's total (robot/<variant>/variant.json).

    python3 tools/mass_breakdown.py v19_sts3032 [more variants]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE = r'''
import json, os, numpy as np
import haetburi_params as P, make_display_model as MD
parts, _, _, _ = MD.build(P.CHAR_HEIGHT, log=lambda *a: None,
                          display_mm=(P.DISPLAY_OUTLINE[1], P.DISPLAY_OUTLINE[2], P.DISPLAY_OUTLINE[0]))
k = P.MESH_SCALE
walls = getattr(P, "SHELL_WALL", {})
rho = P.RHO_PLA * P.RIB_FACTOR                       # g/mm^3 x rib factor
shell = {}
for n, m in parts.items():
    key = {"arm_a": "arms", "arm_b": "arms", "foot_a": "feet", "foot_b": "feet",
           "body_legs": "body + legs"}.get(n, n)
    w = walls.get(n.split("_")[0], P.WALL_MM)
    shell[key] = shell.get(key, 0.0) + m.area * k * k * w * rho / 1000.0
elec = sum(m for m, *_ in getattr(P, "ELECTRONICS_ITEMS", [])) or P.ELECTRONICS_KG
print(json.dumps(dict(shell=shell, walls=dict(walls, default=P.WALL_MM), servo_kg=P.MOTOR_MASS_KG,
                      display_kg=P.DISPLAY_MASS_KG + P.DISPLAY_MOUNT_KG, electronics_kg=elec,
                      head_extra_kg=P.HEAD_EXTRA_KG, servo=P.SERVO)))
'''


def breakdown(v):
    env = dict(os.environ, HAETBURI_VARIANT=v, PYTHONWARNINGS="ignore")
    out = subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=env, capture_output=True, text=True, check=True)
    b = json.loads(out.stdout.strip().splitlines()[-1])
    info = json.load(open(os.path.join(ROOT, "robot", v, "variant.json")))
    n = int(info["summary"]["actuated joints"].split()[0])
    rows = [(f"shell: {name} ({b['walls'].get(name.split(' ')[0].rstrip('s'), b['walls']['default']):.0f} mm)", kg)
            for name, kg in sorted(b["shell"].items(), key=lambda x: -x[1])]
    rows += [(f"servos: {n} x {b['servo']}", n * b["servo_kg"]),
             ("display + mount", b["display_kg"]),
             ("electronics (battery, boards, regulators, wiring)", b["electronics_kg"]),
             ("head extra (fixings)", b["head_extra_kg"])]
    rest = info["mass_kg"] - sum(kg for _, kg in rows)
    rows.append(("rest: brackets, sole plates, screws (total - the above)", rest))
    lines = [f"{v}: {info['mass_kg']:.3f} kg, {info['height_mm']:.0f} mm"]
    lines += [f"  {name:58s} {kg:6.3f} kg  {kg / info['mass_kg'] * 100:4.0f}%" for name, kg in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        print(breakdown(v))
