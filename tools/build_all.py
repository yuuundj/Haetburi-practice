#!/usr/bin/env python3
"""Build, analyse and render every variant, then an overview sheet.

    MUJOCO_GL=glfw python3 tools/build_all.py            # everything
    MUJOCO_GL=glfw python3 tools/build_all.py v4_ankle_in_foot v6_split_round
    MUJOCO_GL=glfw python3 tools/build_all.py --skip-build  # re-render / re-analyse only

Per variant, in robot/<variant>/:
  haetburi.urdf, haetburi_scene.xml, meshes/   the model
  build_report.txt, variant.json               masses, sizes, motor containment
  analysis.txt                                 sim/analyse.py + a headless demo run
  preview.png                                  contact sheet with x-ray views
Overview of all variants: robot/variants_overview.png
"""
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))


def variants_all():
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-c",
                          "import haetburi_params as P; print(' '.join(P.VARIANTS))"],
                         cwd=HERE, env=env, capture_output=True, text=True, check=True)
    return out.stdout.split()


def build(v):
    env = dict(os.environ, HAETBURI_VARIANT=v, PYTHONWARNINGS="ignore")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_model.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"build {v} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def analyse(v):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sim", "analyse.py"), "--variant", v],
                       env=dict(os.environ, PYTHONWARNINGS="ignore"), capture_output=True, text=True)
    text = r.stdout if r.returncode == 0 else r.stdout + r.stderr

    import mujoco
    import demo
    demo.configure(v)
    m = mujoco.MjModel.from_xml_path(demo.SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    z0, fell, peak = d.subtree_com[1][2], None, {}
    while d.time < demo.DURATION:
        demo.drive(m, d, d.time)
        mujoco.mj_step(m, d)
        if fell is None and d.subtree_com[1][2] < 0.75 * z0:
            fell = (round(d.time, 1), demo.label_at(d.time))
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            peak[n] = max(peak.get(n, 0.0), abs(d.actuator_force[i]))
    text += "\ndemo (sim/demo.py, 23.5 s, amplitudes from variant.json)\n"
    text += f"  {'fell at %.1f s during %s' % fell if fell else 'stayed up the whole run'}\n"
    text += "  peak torque, % of stall: " + ", ".join(
        f"{k} {v_ / m.actuator_forcerange[:, 1].max() * 100:.0f}" for k, v_ in peak.items()) + "\n"
    with open(os.path.join(ROOT, "robot", v, "analysis.txt"), "w") as fh:
        fh.write(text)
    return text


def label(img, text, h=30):
    im = Image.fromarray(img)
    g = ImageDraw.Draw(im)
    g.rectangle([0, 0, im.width, h], fill=(18, 20, 28))
    g.text((8, 9), text, fill=(240, 240, 240))
    return np.array(im)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = "--skip-build" in sys.argv
    todo = args or variants_all()
    import render_preview
    columns = []
    for v in todo:
        print(f"== {v}", flush=True)
        if not skip:
            print(build(v).split("\n")[0])
        a = analyse(v)
        print("\n".join(l for l in a.splitlines() if "tip-over" in l or "demo" in l or "stayed" in l
                        or "fell" in l))
        with redirect_stdout(io.StringIO()):
            tiles = render_preview.render(v)
        columns.append(np.vstack([label(tiles[0], v), tiles[5], tiles[3]]))
    if not args:
        sheet = np.hstack(columns)
        im = Image.fromarray(sheet)
        im = im.resize((im.width * 6 // 10, im.height * 6 // 10), Image.LANCZOS)
        out = os.path.join(ROOT, "robot", "variants_overview.png")
        im.save(out)
        print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
